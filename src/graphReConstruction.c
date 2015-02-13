/*
Copyright 2007, 2008 Daniel Zerbino (zerbino@ebi.ac.uk)

    This file is part of Velvet.

    Velvet is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Velvet is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Velvet; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "defines.h"
#include "globals.h"
#include "graph.h"
#include "passageMarker.h"
#include "readSet.h"
#include "tightString.h"
#include "recycleBin.h"
#include "utility.h"
#include "kmer.h"
#include "kmerOccurenceTable.h"
#include "roadMap.h"

#define ADENINE 0
#define CYTOSINE 1
#define GUANINE 2
#define THYMINE 3


//////////////////////////////////////////////////////////
// Node Locking
//////////////////////////////////////////////////////////

#ifdef _OPENMP

/* Array of per-node locks */

static omp_lock_t *nodeLocks = NULL;

static void
createNodeLocks(Graph *graph)
{
	IDnum nbNodes;
	IDnum nodeIndex;

	nbNodes = nodeCount(graph) + 1;
	if (nodeLocks)
		free (nodeLocks);
	nodeLocks = mallocOrExit(nbNodes, omp_lock_t);

	#pragma omp parallel for
	for (nodeIndex = 0; nodeIndex < nbNodes; nodeIndex++)
		omp_init_lock(nodeLocks + nodeIndex);
}

static inline void lockNode(Node *node)
{
	IDnum nodeID = getNodeID(node);

	if (nodeID < 0)
		nodeID = -nodeID;
	omp_set_lock (nodeLocks + nodeID);
}

/* Assumes node is already locked */
static inline void lockTwoNodes(Node *node, Node *node2)
{
	IDnum nodeID = getNodeID(node);
	IDnum node2ID = getNodeID(node2);

	if (nodeID < 0)
		nodeID = -nodeID;
	if (node2ID < 0)
		node2ID = -node2ID;

	if (nodeID == node2ID)
		return;

	/* Lock lowest ID first to avoid deadlocks */
	if (nodeID < node2ID)
	{
		omp_set_lock (nodeLocks + node2ID);
	}
	else if (!omp_test_lock (nodeLocks + node2ID))
	{
		omp_unset_lock (nodeLocks + nodeID);
		omp_set_lock (nodeLocks + node2ID);
		omp_set_lock (nodeLocks + nodeID);
	}
}

static inline void unLockTwoNodes(Node *node, Node *node2)
{
	IDnum nodeID = getNodeID(node);
	IDnum node2ID = getNodeID(node2);

	if (nodeID < 0)
		nodeID = -nodeID;
	if (node2ID < 0)
		node2ID = -node2ID;

	omp_unset_lock (nodeLocks + nodeID);
	if (nodeID != node2ID)
		omp_unset_lock (nodeLocks + node2ID);
}

static inline void unLockNode(Node *node)
{
	IDnum nodeID = getNodeID(node);

	if (nodeID < 0)
		nodeID = -nodeID;
	omp_unset_lock (nodeLocks + nodeID);
}

#endif

//////////////////////////////////////////////////////////
// Node Lists
//////////////////////////////////////////////////////////
typedef struct smallNodeList_st SmallNodeList;

struct smallNodeList_st {
	Node *node;
	SmallNodeList *next;
} ATTRIBUTE_PACKED;

static RecycleBin *smallNodeListMemory = NULL;

#define BLOCKSIZE 1000

#ifdef _OPENMP
static void initSmallNodeListMemory(void)
{
	int n = omp_get_max_threads();

#pragma omp critical
	{
		if (smallNodeListMemory == NULL)
			smallNodeListMemory = newRecycleBinArray(n, sizeof(SmallNodeList), BLOCKSIZE);
	}
}
#endif

static SmallNodeList *allocateSmallNodeList()
{
#ifdef _OPENMP
#ifdef DEBUG
	if (smallNodeListMemory == NULL)
	{
		velvetLog("The memory for small nodes seems uninitialised, "
				"this is probably a bug, aborting.\n");
		abort();
	}
#endif
	return allocatePointer(getRecycleBinInArray(smallNodeListMemory,
				omp_get_thread_num()));
#else
	if (smallNodeListMemory == NULL)
		smallNodeListMemory = newRecycleBin(sizeof(SmallNodeList), BLOCKSIZE);

	return allocatePointer(smallNodeListMemory);
#endif
}

static void deallocateSmallNodeList(SmallNodeList * smallNodeList)
{
#ifdef _OPENMP
	deallocatePointer(getRecycleBinInArray(smallNodeListMemory,
					       omp_get_thread_num()),
			  smallNodeList);
#else
	deallocatePointer(smallNodeListMemory, smallNodeList);
#endif
}

static void destroySmallNodeListMemmory(void)
{
	if (smallNodeListMemory != NULL)
	{
#ifdef _OPENMP
		destroyRecycleBinArray(smallNodeListMemory);
#else
		destroyRecycleBin(smallNodeListMemory);
#endif
		smallNodeListMemory = NULL;
	}
}

static inline void memorizeNode(Node * node, SmallNodeList ** nodePile)
{
	SmallNodeList *list = allocateSmallNodeList();
	list->node = node;
	list->next = *nodePile;
	*nodePile = list;
#ifndef _OPENMP
   	setSingleNodeStatus(node, true);
#endif
}

static inline boolean isNodeMemorized(Node * node, SmallNodeList * nodePile)
{
#ifdef _OPENMP
	/* SF TODO There must be a faster way to do this: bit mask, hash table, tree, ... ? */
	SmallNodeList * list;

	for (list = nodePile; list; list = list->next)
		if (list->node == node)
			return true;

	return false;
#else
	return getNodeStatus(node);
#endif
}

static void unMemorizeNodes(SmallNodeList ** nodePile)
{
	SmallNodeList * list;

	while (*nodePile) {
		list = *nodePile;
		*nodePile = list->next;
#ifndef _OPENMP
   		setSingleNodeStatus(list->node, false);
#endif
		deallocateSmallNodeList(list);
	}
}

///////////////////////////////////////////////////////////
// Reference Mappings
///////////////////////////////////////////////////////////
typedef struct referenceMapping_st ReferenceMapping;

struct referenceMapping_st {
	IDnum referenceStart;
	IDnum nodeStart;
	IDnum length;
	IDnum referenceID;
	IDnum nodeID;
} ATTRIBUTE_PACKED;

static IDnum countMappings(char * preGraphFilename) {
	FILE *file = fopen(preGraphFilename, "r");
	const int maxline = MAXLINE;
	char line[MAXLINE];
	IDnum count = 0;

	// Go past NODE blocks
	while(fgets(line, maxline, file))
		if (line[0] == 'S')
			break;

	// Count relevant lines
	while(fgets(line, maxline, file))
		if (line[0] != 'S')
			count++;

	fclose(file);
	return count;
}

static ReferenceMapping * recordReferenceMappings(char * preGraphFilename, IDnum arrayLength) {
	ReferenceMapping * mappings = callocOrExit(arrayLength, ReferenceMapping);
	FILE *file = fopen(preGraphFilename, "r");
	const int maxline = MAXLINE;
	char line[MAXLINE];
	ReferenceMapping * current = mappings;
	IDnum referenceID;
	long long_var;
	long long coord1, coord2, coord3; 
	
	// Go past NODE blocks
	while(fgets(line, maxline, file))
		if (line[0] == 'S')
			break;

	sscanf(line, "SEQ\t%li\n", &long_var);
	referenceID = long_var;

	// Go relevant lines
	while(fgets(line, maxline, file)) {
		if (line[0] != 'S') {
			sscanf(line, "%li\t%lli\t%lli\t%lli\n", &long_var, &coord1, &coord2, &coord3);
			current->referenceID = referenceID;
			current->nodeID = long_var;
			current->nodeStart = coord1;
			current->referenceStart = coord2;
			current->length = coord3;
			current++;
		} else {
			sscanf(line, "SEQ\t%li\n", &long_var);
			referenceID = long_var;
		} 
	}

	fclose(file);
	return mappings;
}

static int compareRefMaps(const void * ptrA, const void * ptrB) {
	ReferenceMapping * A = (ReferenceMapping *) ptrA;
	ReferenceMapping * B = (ReferenceMapping *) ptrB;

	if (A->referenceID > B->referenceID) 
		return 1;
	else if (A->referenceID < B->referenceID)
		return -1;
	else {
		if (A->referenceStart >= B->referenceStart + B->length)
			return 1;
		else if (A->referenceStart + A->length <= B->referenceStart)
			return -1;
		else 
			return 0;
	}
}

static ReferenceMapping * computeReferenceMappings(char * preGraphFilename, ReadSet * reads, Coordinate * referenceMappingLength, IDnum * referenceCount) {
	IDnum index;
	ReferenceMapping * referenceMappings;

	for(index = 0; index < reads->readCount && reads->categories[index] == 2 * CATEGORIES + 2; index++)
		(*referenceCount)++;

	if (*referenceCount == 0) {
		*referenceMappingLength = 0;
		return NULL;
	}

	*referenceMappingLength = countMappings(preGraphFilename);
	
	if (*referenceMappingLength == 0)
		return NULL;

	referenceMappings = recordReferenceMappings(preGraphFilename, *referenceMappingLength); 
	qsort(referenceMappings, *referenceMappingLength, sizeof(ReferenceMapping), compareRefMaps);

	return referenceMappings;
}

static ReferenceMapping * findReferenceMapping(IDnum seqID, Coordinate refCoord, ReferenceMapping * referenceMappings, Coordinate referenceMappingCount) {
	IDnum positive_seqID;
	Coordinate leftIndex = 0;
	Coordinate rightIndex = referenceMappingCount - 1;
	Coordinate middleIndex;
	ReferenceMapping refMap;
	int comparison;

	if (seqID > 0)
		positive_seqID = seqID;
	else
		positive_seqID = -seqID;

	refMap.referenceID = positive_seqID;
	refMap.referenceStart = refCoord;
	refMap.length = 1;
	refMap.nodeStart = 0;
	refMap.nodeID = 0;

	if (compareRefMaps(&(referenceMappings[leftIndex]), &refMap) == 0)
		return &(referenceMappings[leftIndex]);
	if (compareRefMaps(&(referenceMappings[rightIndex]), &refMap) == 0)
		return &(referenceMappings[rightIndex]);

	while (true) {
		middleIndex = (rightIndex + leftIndex) / 2;
		comparison = compareRefMaps(&(referenceMappings[middleIndex]), &refMap);

		if (leftIndex >= rightIndex)
			return NULL;
		else if (comparison == 0)
			return &(referenceMappings[middleIndex]);
		else if (leftIndex == middleIndex)
			return NULL;
		else if (comparison > 0)
			rightIndex = middleIndex;
		else
			leftIndex = middleIndex;
	}
}
 
///////////////////////////////////////////////////////////
// Node Mask
///////////////////////////////////////////////////////////

typedef struct nodeMask_st NodeMask;

struct nodeMask_st {
	IDnum nodeID;
	IDnum start;
	IDnum finish;
} ATTRIBUTE_PACKED;

static int compareNodeMasks(const void * ptrA, const void * ptrB) {
	NodeMask * A = (NodeMask *) ptrA;
	NodeMask * B = (NodeMask *) ptrB;
	
	if (A->nodeID < B->nodeID)
		return -1;
	else if (A->nodeID > B->nodeID)
		return 1;
	else {
		if (A->start < B->start)
			return -1;
		else if (A->start > B->start)
			return 1;
		else 
			return 0;
	}
}

static NodeMask * computeNodeMasks(ReferenceMapping * referenceMappings, Coordinate arrayLength, Graph * graph) {
	NodeMask * nodeMasks;
	NodeMask * currentMask;
	ReferenceMapping * currentMapping = referenceMappings;
	Coordinate index;

	if (referenceMappings == NULL)
		return NULL;

	nodeMasks = callocOrExit(arrayLength, NodeMask);
	currentMask = nodeMasks;

	for (index = 0; index < arrayLength; index++) {
		if (currentMapping->nodeID > 0) {
			currentMask->nodeID = currentMapping->nodeID;
		} else {
			currentMask->nodeID = -currentMapping->nodeID;
		}
		currentMask->start = currentMapping->nodeStart;
		currentMask->finish = currentMapping->nodeStart + currentMapping->length;
		currentMask++;
		currentMapping++;
	}

	qsort(nodeMasks, arrayLength, sizeof(NodeMask), compareNodeMasks);

	return nodeMasks;
}

///////////////////////////////////////////////////////////
// Process
///////////////////////////////////////////////////////////

static KmerOccurenceTable *referenceGraphKmers(char *preGraphFilename,
					       short int accelerationBits, Graph * graph, boolean double_strand, NodeMask * nodeMasks, Coordinate nodeMaskCount)
{
	FILE *file = fopen(preGraphFilename, "r");
	const int maxline = MAXLINE;
	char line[MAXLINE];
	char c;
	int wordLength;
	Coordinate lineLength, kmerCount;
	Kmer word;
	Kmer antiWord;
	KmerOccurenceTable *kmerTable;
	#if TEST_PARALLEL_KMER_FILLTABLE
        KmerOccurenceTable *kmerTablePARALLEL;  //debug
        #endif

	IDnum index;
	IDnum nodeID = 0;
	Nucleotide nucleotide;
	NodeMask * nodeMask = nodeMasks; 
	Coordinate nodeMaskIndex = 0;

        int debug = 0;

	if (file == NULL)
		exitErrorf(EXIT_FAILURE, true, "Could not open %s", preGraphFilename);

           
	// Count kmers
	velvetLog("Scanning pre-graph file %s for k-mers\n",
		  preGraphFilename);


	// First  line
	if (!fgets(line, maxline, file))
		exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
	sscanf(line, "%*i\t%*i\t%i\n", &wordLength);

	kmerTable = newKmerOccurenceTable(accelerationBits, wordLength);

        #if TEST_PARALLEL_KMER_FILLTABLE
	if(1) kmerTablePARALLEL = newKmerOccurenceTable(accelerationBits, wordLength);  // debug
        #endif

	// Read nodes
	if (!fgets(line, maxline, file))
		exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
	kmerCount = 0;

        #if !PARALLEL_KMERCOUNT
	// original code 
	while (line[0] == 'N') {
		lineLength = 0;
		while ((c = getc(file)) != EOF && c != '\n')
			lineLength++;
		kmerCount += lineLength - wordLength + 1;
		if (fgets(line, maxline, file) == NULL)
			break;
	}
	velvetLog("%li kmers found\n", (long) kmerCount);

        #else   // PARALLEL_KMERCOUNT
               velvetLog(" doing parallel kmercount\n");
               // determine size of input file
               fseek(file,0L, SEEK_END);
               off_t fsize = ftell(file);
               //velvetLog(" input file size: %s bytes: %d\n",preGraphFilename,fsize);
               (void) fseek(file, 0L, SEEK_SET);   // position at begining
               int numThreads = omp_get_num_threads();

		//  split file into partitions, count kmerCount in each task, 
		//  with reduction at the end for the total kmerCount
               kmerCount = 0;
	       int itask=0;

	       #pragma omp parallel
	       {
		  int numThreads = omp_get_num_threads(); // obeys OMP_NUM_THREADS env variable
		  int ntask = numThreads * 1;   // use higher factor for better loadbalancing

		  #pragma omp single
		  {
		     int i;
		     for(i=0;i< ntask; i++) {
		       #pragma omp task
		       {
			  FILE *t_file =  fopen(preGraphFilename, "r");
			  if (t_file == NULL)  exit(EXIT_FAILURE);
			  int my_itask;
			  #pragma omp atomic capture
                             my_itask=itask++; 
			  
			  // ibeg and iend are byte offesets into the file for each partition
                          // should fix for rounding/truncation, and examine getline return codes
			  ssize_t ibeg = (fsize/ntask) * my_itask; 
			  ssize_t iend = (fsize/ntask) * (my_itask+1);
			  ssize_t filePoint = ibeg;
			  char *line = NULL;
			  size_t bufsize; 
			  ssize_t len = 0;
			  ssize_t ssumt = 0;

			  fseek(t_file, ibeg, SEEK_SET);  //positon to begining of partition

			  if(my_itask==0) {  // skip first line if first task
			     len = getline(&line, &bufsize, t_file); filePoint += (len ); 
                             if (len == -1)  exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
			  }
			  while (filePoint < iend ) {  // this code assumes node/kmer records occur in pairs
			     len = getline(&line, &bufsize, t_file); filePoint += (len ); 
                             if (len == -1)  exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");

			     if (line[0] == 'N') {
                                int xlineLength = 0;
				len = getline(&line, &bufsize, t_file); filePoint += (len ); 
                                if (len == -1)  exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
                                ssumt += ( len - 1)  - wordLength + 1;
			     }
			  }
			  if (line) free(line);
			  close(t_file);
                          #pragma omp atomic update
			  kmerCount += ssumt;
		       } // closing bracket: omp task
		     } // closing bracket: for  
		  } // closing bracket: omp single
                  #pragma omp taskwait
	       } // closing bracket: omp parallel

	       velvetLog("%li kmers found\n", (long) kmerCount);

	 #endif   // PARALLEL_KMERCOUNT

	       for(nodeMaskIndex = 0; nodeMaskIndex < nodeMaskCount; nodeMaskIndex++) {
		       kmerCount -= nodeMasks[nodeMaskIndex].finish -
			       nodeMasks[nodeMaskIndex].start;
	       }
	       nodeMaskIndex = 0;
	       fclose(file);

	       // Create table
	       allocateKmerOccurences(kmerCount + 1, kmerTable);  // added 1 for comfort, otherwise getting SEGV in the FILLTABLE code
	       printf("   kmerTable:%x \n", kmerTable);

               #if TEST_PARALLEL_KMER_FILLTABLE
	       if(1) allocateKmerOccurences(kmerCount + 1, kmerTablePARALLEL);  // added 1 for comfort, otherwise getting SEGV in the FILLTABLE code
	       printf("   kmerTable:%x \n", kmerTablePARALLEL);
	       #endif



	       // Fill table

#if  !PARALLEL_KMER_FILLTABLE  || TEST_PARALLEL_KMER_FILLTABLE

        // this is the original code

	// Fill table
	file = fopen(preGraphFilename, "r");
	if (file == NULL)
		exitErrorf(EXIT_FAILURE, true, "Could not open %s", preGraphFilename);

	if (!fgets(line, maxline, file))
		exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");

	// Read nodes
	
	if (!fgets(line, maxline, file))
		exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
	while (line[0] == 'N') {
		nodeID++;

		// Fill in the initial word : 
		clearKmer(&word);
		clearKmer(&antiWord);

		for (index = 0; index < wordLength - 1; index++) {
			c = getc(file);
			if (c == 'A')
				nucleotide = ADENINE;
			else if (c == 'C')
				nucleotide = CYTOSINE;
			else if (c == 'G')
				nucleotide = GUANINE;
			else if (c == 'T')
				nucleotide = THYMINE;
			else if (c == '\n')
				exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
			else
				nucleotide = ADENINE;
				

			pushNucleotide(&word, nucleotide);
			if (double_strand) {
#ifdef COLOR
				reversePushNucleotide(&antiWord, nucleotide);
#else
				reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
			}
		}

		// Scan through node
		index = 0;
		while((c = getc(file)) != '\n' && c != EOF) {
			if (c == 'A')
				nucleotide = ADENINE;
			else if (c == 'C')
				nucleotide = CYTOSINE;
			else if (c == 'G')
				nucleotide = GUANINE;
			else if (c == 'T')
				nucleotide = THYMINE;
			else
				nucleotide = ADENINE;

			pushNucleotide(&word, nucleotide);
			if (double_strand) {
#ifdef COLOR
				reversePushNucleotide(&antiWord, nucleotide);
#else
				reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
			}

			// Update mask if necessary 
			if (nodeMask) { 
				if (nodeMask->nodeID < nodeID || (nodeMask->nodeID == nodeID && index >= nodeMask->finish)) {
					if (++nodeMaskIndex == nodeMaskCount) 
						nodeMask = NULL;
					else 
						nodeMask++;
				}
			}

			// Check if not masked!
			if (nodeMask) { 
				if (nodeMask->nodeID == nodeID && index >= nodeMask->start && index < nodeMask->finish) {
					index++;
					continue;
				} 			
			}

			if (!double_strand || compareKmers(&word, &antiWord) <= 0)
				recordKmerOccurence(&word, nodeID, index, kmerTable);
			else
				recordKmerOccurence(&antiWord, -nodeID, getNodeLength(getNodeInGraph(graph, nodeID)) - 1 - index, kmerTable);

			index++;
		}

		if (fgets(line, maxline, file) == NULL)
			break;
	}

	fclose(file);
        fprintf(stdout,"  kmerOccuranceIndex:%d \n", kmerTable->kmerOccurenceIndex);
               velvetLog(" --- done serial fill table \n");

        #if TEST_PARALLEL_KMER_FILLTABLE
        {
            velvetLog("  --- comparing tables\n");
            long i;
            KmerOccurence * kmerOccurence = kmerTable->kmerTable;
            printf(" kmerCount: %ld \n", kmerCount);
            for (i = 0;i< 20;i++ ) {
                fprintf(stdout,"  %ld  nodeid:%d position:%d \n", i, kmerOccurence->nodeID,kmerOccurence->position );
                kmerOccurence++;
            }
        }
        #endif

#endif   // !PARALLEL_KMER_FILLTABLE !! TEST_PARALLEL_KMER_FILLTABLE

#if PARALLEL_KMER_FILLTABLE || TEST_PARALLEL_KMER_FILLTABLE
            {
               velvetLog(" --- using parallel fill table \n");
               int itask = 0;
               int64_t xnn=0;

	       FILE *file = fopen(preGraphFilename, "r");
               fseek(file,0L, SEEK_END);
               long fsize = ftell(file);
               velvetLog(" input file size: %s bytes: %d\n",preGraphFilename,fsize);
               (void) fseek(file, 0L, SEEK_SET);   // position at begining
               fclose(file);

               #pragma omp parallel
	       {
                  int numThreads = omp_get_num_threads();
                  int ntask;
                  if (fsize >  100000)   // only split if file is big
                      ntask = numThreads * 1; // number of tasks to use, higher for load balancing
                  else
                      ntask = 1;


                  #pragma omp single
		  {
                     velvetLog(" --- using ntasks:%d\n", ntask);
		     long i ;
                     long *partitionTable =  malloc( (ntask+2)  * sizeof( long) ); // need to free
                     long ppp[100];

                     for (i=0;i< ntask ; i++) {
                           partitionTable[i] =  (fsize/ntask) * (i+1);
                           ppp[i] = (fsize/ntask) * (i+1);
                     }
                     partitionTable[ntask-1] = fsize ; // insure last partition is max fsize
                     ppp[ntask-1] = fsize ; // insure last partition is max fsize

                     //for debug
                     #if DEBUG
                     for (i=0;i< ntask ; i++) {
                           printf("     partition:%d  %ld  %ld\n", i, partitionTable[i], fsize);
                     }
                     for (i=0;i< ntask ; i++) {
                           printf("     ppp:%d  %ld  %ld\n", i, ppp[i], fsize);
                     }
                     #endif


		     for (i=0;i< ntask; i++) {
                       #pragma omp task
		       {
	                  FILE *t_file = fopen(preGraphFilename, "r");
	                  if (t_file == NULL)
		              exitErrorf(EXIT_FAILURE, true, "Could not open %s", preGraphFilename);

			   long my_itask;

                           #pragma omp atomic capture
                              my_itask=itask++; 

                           //velvetLog(" T%ld ... opened %s \n",my_itask,preGraphFilename);

                           char c;
			   Kmer word; 
                           Kmer antiWord; 
                           IDnum index; 
                           IDnum nodeID = 0;

			   Nucleotide nucleotide;
			   NodeMask * nodeMask = nodeMasks;
                           Coordinate nodeMaskIndex = 0;


			   // ibeg and iend are byte offesets into the file for each partition
                           long ibeg =0; 
                           //if (my_itask > 0) ibeg = partitionTable[my_itask-1] ;
                           //long iend = partitionTable[my_itask];
                           if (my_itask > 0) ibeg = ppp[my_itask-1] ;
                           long iend = ppp[my_itask];

                           //fprintf(stdout," T%ld  ibeg,iend %ld, %ld \n",my_itask,ibeg, iend);
                           //if (my_itask > 0) fprintf(stdout," T%ld  ps,pe %ld, %ld   %ld,%ld\n",my_itask, partitionTable[my_itask-1],partitionTable[my_itask], ibeg, iend);
                           //if (my_itask > 0) fprintf(stdout," T%ld  ps,pe %ld,%ld   %ld,%ld\n",my_itask, ppp[my_itask-1],ppp[my_itask], ibeg, iend);

			   char *line = NULL;
                           size_t bufsize; 
                           
                           long filePoint = ibeg;
			   ssize_t len = 0;
         
                           fseek(t_file, ibeg, SEEK_SET);  // seek to beginging of partition
                           len = getline(&line, &bufsize, t_file); filePoint += (len ); 
                           if( len == -1) exitErrorf(EXIT_FAILURE, true, " T%ld PreGraph file reached end - A\n", my_itask);
			   int cursor = 0;
                           int flag = 1;

		           while (1) {

                              // testing for 'N' allows for skipping fist line and any fragment of line
                              // that may exist at the begining of a partition
			      if(line[0] == 'N')  {
                                 //this code relys on the nodeid being in the PreGrpaph file
			         //read nodeid from NODE record, skipping over the 4 chars "NODE"  
                                 // scanf's are expensive, we do only once at the begining of 
                                 // partition as the nodeID's are in sequence in the PreGraph file

                                    #pragma omp atomic
                                    xnn++;


                                 if (flag == 1) { // at beginging, sscanf the nodeID
                                    sscanf(line + 4, "%ld", &nodeID); 
                                    flag = 0;
                                 } else {
                                    nodeID++;   // no need to scan, just increment
                                 }
                                 //printf("  nodeID:%d  \n",nodeID);
                                    
                                 len = getline(&line, &bufsize, t_file); filePoint += (len ); 
				 if ( len == -1) exitErrorf(EXIT_FAILURE, true, " T%ld PreGraph file reached end - B\n", my_itask);
                                 cursor=0;

		                 // Fill in the initial word : 
		                 clearKmer(&word);
		                 clearKmer(&antiWord);
      
		                 for (index = 0; index < wordLength - 1; index++) {
                                    c = line[cursor++];
			            if (c == 'A') nucleotide = ADENINE;
			            else if (c == 'C') nucleotide = CYTOSINE;
			            else if (c == 'G') nucleotide = GUANINE;
			            else if (c == 'T') nucleotide = THYMINE;
			            else if (c == '\n') 
				       exitErrorf(EXIT_FAILURE, true, "PreGraph file incomplete");
			            else nucleotide = ADENINE;
				
			            pushNucleotide(&word, nucleotide);

			            if (double_strand) {
                                       #ifdef COLOR
		                       reversePushNucleotide(&antiWord, nucleotide);
                                       #else
			               reversePushNucleotide(&antiWord, 3 - nucleotide);
                                       #endif
			            }
		                 }
		        
		                 // Scan through node
		                 index = 0;
                                 while(  (c = line[cursor++]) != '\n'  && c != EOF )
                                 {
                                    if (c == 'A') nucleotide = ADENINE;
                                    else if (c == 'C') nucleotide = CYTOSINE;
			            else if (c == 'G') nucleotide = GUANINE;
			            else if (c == 'T') nucleotide = THYMINE;
			            else nucleotide = ADENINE;
   
			            pushNucleotide(&word, nucleotide);
			            if (double_strand) {
                                       #ifdef COLOR
			               reversePushNucleotide(&antiWord, nucleotide);
                                       #else
			               reversePushNucleotide(&antiWord, 3 - nucleotide);
                                       #endif
			            }

			            // Update mask if necessary 
			            if (nodeMask) { 
                                       // this code path has not been tested if PARALLEL_KMER_FILLTABLE defined
                                       exitErrorf(EXIT_FAILURE, true, " nodeMask code path not tested ");

				       if (nodeMask->nodeID < nodeID || (nodeMask->nodeID == nodeID && index >= nodeMask->finish)) {
					  if (++nodeMaskIndex == nodeMaskCount) 
					      nodeMask = NULL;
					  else 
					      nodeMask++;
				       }
				    }
				    // Check if not masked!
			            if (nodeMask) { 
                                       // this code path has not been tested if PARALLEL_KMER_FILLTABLE defined
				       if (nodeMask->nodeID == nodeID && index >= nodeMask->start && index < nodeMask->finish) {
					  index++;
					  continue;
				       } 			
				    }

                                    #if !TEST_PARALLEL_KMER_FILLTABLE
				    if (!double_strand || compareKmers(&word, &antiWord) <= 0)
				       fast_recordKmerOccurence(&word, nodeID, index, kmerTable);
				    else
				       fast_recordKmerOccurence(&antiWord, -nodeID, getNodeLength(getNodeInGraph(graph, nodeID)) - 1 - index, kmerTable);
                                    #else 
				    if (!double_strand || compareKmers(&word, &antiWord) <= 0)
				       fast_recordKmerOccurence(&word, nodeID, index, kmerTablePARALLEL);
				    else
				       fast_recordKmerOccurence(&antiWord, -nodeID, getNodeLength(getNodeInGraph(graph, nodeID)) - 1 - index, kmerTablePARALLEL);
                                    #endif

				    index++;
				 }

	                      } // closing bracket: if(line[0] == 'N')

			      if (filePoint > iend - 1) break;

                              len = getline(&line, &bufsize, t_file); filePoint += (len );
                              if (len == -1) break;

                           } // closing bracket: while (1)
              
                           if (line) free(line);  
	                   fclose(t_file);

                        } // closig bracket: omp task
                     } // closing bracket: for
                     free(partitionTable);

                  } // closing bracket: omp single
                  #pragma omp taskwait
               } // closing bracket: omp parallel
               velvetLog("  --- xnn:%ld \n", xnn);

            }
        fprintf(stdout,"  kmerOccuranceIndex   :%d \n", kmerTable->kmerOccurenceIndex);
        #if TEST_PARALLEL_KMER_FILLTABLE
        fprintf(stdout,"  kmerOccuranceIndex PARALLEL  :%d \n", kmerTablePARALLEL->kmerOccurenceIndex);
        #endif

        velvetLog("  --- done with parallel fill table\n");
#endif   // PARALLEL_KMER_FILLTABLE || kmerTablePARALLEL

        velvetLog("  --- done with fill table\n");

	// Sort table
	sortKmerOccurenceTable(kmerTable);
        velvetLog("  --- done with sorts table\n");
        if(0) {
            velvetLog("  --- after sort table kmerTable\n");
            long i;
            KmerOccurence * kmerOccurence = kmerTable->kmerTable;
            printf(" kmerCount: %ld \n", kmerCount);
            for (i = 0;i< 20;i++ ) {
                fprintf(stdout,"  %ld  nodeid:%d position:%d \n", i, kmerOccurence->nodeID,kmerOccurence->position );
                kmerOccurence++;
            }
        }


        #if TEST_PARALLEL_KMER_FILLTABLE
        if(1) {
            velvetLog("  --- before sort table kmerTablePARALLEL\n");
            long i;
            KmerOccurence * kmerOccurence = kmerTablePARALLEL->kmerTable;
            printf(" kmerCount: %ld \n", kmerCount);
            for (i = 0;i< 20;i++ ) {
                fprintf(stdout,"  %ld  nodeid:%d position:%d \n", i, kmerOccurence->nodeID,kmerOccurence->position );
                kmerOccurence++;
            }
        }
	if(1) sortKmerOccurenceTable(kmerTablePARALLEL);
        if(1) {
            velvetLog("  --- after sort table kmerTablePARALLEL\n");
            long i;
            KmerOccurence * kmerOccurence = kmerTablePARALLEL->kmerTable;
            printf(" kmerCount: %ld \n", kmerCount);
            for (i = 0;i< 20;i++ ) {
                fprintf(stdout,"  %ld  nodeid:%d position:%d \n", i, kmerOccurence->nodeID,kmerOccurence->position );
                kmerOccurence++;
            }
        }

        if(1) { 
            velvetLog("  --- comparing tables\n");
            long i;
            long nerrors = 0;
            KmerOccurence * kmerOccurence = kmerTable->kmerTable;
            KmerOccurence * kmerOccurencePARALLEL = kmerTablePARALLEL->kmerTable;
            printf(" kmerCount: %ld \n", kmerCount);
            for (i = 0;i< kmerCount + 1;i++ ) {
                //fprintf(stdout,"  %d  nodeid:%d position:%d \n", i, kmerOccurence->nodeID,kmerOccurence->position );

                if(kmerOccurence->nodeID != kmerOccurencePARALLEL->nodeID ) printf("  %ld  nodeid:%d xxxid:%d \n", i, kmerOccurence->nodeID,kmerOccurencePARALLEL->nodeID );
		if(kmerOccurence->position != kmerOccurencePARALLEL->position ) printf("  %ld  position:%d xxxposition:%d \n", i, kmerOccurence->position,kmerOccurencePARALLEL->position );

                if ( 0 != compareKmers(&(kmerOccurence->kmer), &(kmerOccurencePARALLEL->kmer) ) ) {
                    printf("  kmer not equal %ld  nodeid:%d xxxid:%d \n", i, kmerOccurence->nodeID,kmerOccurencePARALLEL->nodeID );
                    nerrors++;
                }
                
                kmerOccurence++;
                kmerOccurencePARALLEL++;
	    }
	    velvetLog("  --- done with comparing tables  errors: %ld \n", nerrors);
         }
         free(kmerTablePARALLEL);
         #endif
	
        #if !PARALLEL_KMER_FILLTABLE || TEST_PARALLEL_KMER_FILLTABLE
        velvetLog("  --- using serially constructed tables\n");
        #endif
        #if PARALLEL_KMER_FILLTABLE 
        velvetLog("  --- using parallely  constructed tables\n");
        #endif

	return kmerTable;
}

static void ghostThreadSequenceThroughGraph(TightString * tString,
					    KmerOccurenceTable *
					    kmerTable, Graph * graph,
					    IDnum seqID, Category category,
					    boolean readTracking,
					    boolean double_strand,
					    ReferenceMapping * referenceMappings,
					    Coordinate referenceMappingCount,
					    IDnum refCount,
					    Annotation * annotations,
					    IDnum annotationCount,
					    boolean second_in_pair)
{
	Kmer word;
	Kmer antiWord;
	Coordinate readNucleotideIndex;
	KmerOccurence *kmerOccurence;
	int wordLength = getWordLength(graph);
	Nucleotide nucleotide;
	IDnum refID;
	Coordinate refCoord;
	ReferenceMapping * refMap = NULL;
	Coordinate uniqueIndex = 0;
	Coordinate annotIndex = 0;
	IDnum annotCount = 0;
	boolean reversed;
	SmallNodeList * nodePile = NULL;
	Annotation * annotation = annotations;

	Node *node = NULL;
	Node *previousNode = NULL;

	// Neglect any read which will not be short paired
	if ((!readTracking && category % 2 == 0)
	    || category / 2 >= CATEGORIES)
		return;

	// Neglect any string shorter than WORDLENGTH :
	if (getLength(tString) < wordLength)
		return;

	// Verify that all short reads are reasonnably short
	if (getLength(tString) > USHRT_MAX) {
		velvetLog("Short read of length %lli, longer than limit %i\n",
			  (long long) getLength(tString), SHRT_MAX);
		velvetLog("You should better declare this sequence as long, because it genuinely is!\n");
		exit(1);
	}

	clearKmer(&word);
	clearKmer(&antiWord);

	// Fill in the initial word :
	for (readNucleotideIndex = 0;
	     readNucleotideIndex < wordLength - 1; readNucleotideIndex++) {
		nucleotide = getNucleotide(readNucleotideIndex, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}
	}

	// Go through sequence
	while (readNucleotideIndex < getLength(tString)) {
		// Shift word:
		nucleotide = getNucleotide(readNucleotideIndex++, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}

		// Update annotation if necessary
		if (annotCount < annotationCount && annotIndex == getAnnotationLength(annotation)) {
			annotation = getNextAnnotation(annotation);
			annotCount++;
			annotIndex = 0;
		}

		// Search for reference mapping
 		if (annotCount < annotationCount && uniqueIndex >= getPosition(annotation) && getAnnotSequenceID(annotation) <= refCount && getAnnotSequenceID(annotation) >= -refCount) {
			refID = getAnnotSequenceID(annotation);
			if (refID > 0)
				refCoord = getStart(annotation) + annotIndex;
			else
				refCoord = getStart(annotation) - annotIndex;
			
			refMap = findReferenceMapping(refID, refCoord, referenceMappings, referenceMappingCount);
			// If success
			if (refMap) {
				if (refID > 0) 
					node = getNodeInGraph(graph, refMap->nodeID);
				else
					node = getNodeInGraph(graph, -refMap->nodeID);
			} else  {
				node = NULL;
				if (previousNode)
					break;
			}
		}
		// if not.. look in table
		else {
			reversed = false;
			if (double_strand) {
				if (compareKmers(&word, &antiWord) <= 0) {
					kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
								       kmerTable);
				} else { 
					kmerOccurence =
					       findKmerInKmerOccurenceTable(&antiWord,
						kmerTable);
					reversed = true;
				}
			} else {
				if (!second_in_pair) {
					kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
								       kmerTable);
				} else { 
					kmerOccurence =
					       findKmerInKmerOccurenceTable(&antiWord,
						kmerTable);
					reversed = true;
				}
			}
			
			if (kmerOccurence) {
				if (!reversed) 
					node = getNodeInGraph(graph, getKmerOccurenceNodeID(kmerOccurence));
				else
					node = getNodeInGraph(graph, -getKmerOccurenceNodeID(kmerOccurence));
			} else {
				node = NULL;
				if (previousNode) 
					break;
			}

		}

		if (annotCount < annotationCount && uniqueIndex >= getPosition(annotation))
			annotIndex++;
		else
			uniqueIndex++;

		previousNode = node;

		// Fill in graph
		if (node && !isNodeMemorized(node, nodePile))
		{
#ifdef _OPENMP
			lockNode(node);
#endif
			incrementReadStartCount(node, graph);
#ifdef _OPENMP
			unLockNode(node);
#endif
			memorizeNode(node, &nodePile);
		}
	}

	unMemorizeNodes(&nodePile);
}

static void threadSequenceThroughGraph(TightString * tString,
				       KmerOccurenceTable * kmerTable,
				       Graph * graph,
				       IDnum seqID, Category category,
				       boolean readTracking,
				       boolean double_strand,
				       ReferenceMapping * referenceMappings,
				       Coordinate referenceMappingCount,
				       IDnum refCount,
				       Annotation * annotations,
				       IDnum annotationCount,
				       boolean second_in_pair)
{
	Kmer word;
	Kmer antiWord;
	Coordinate readNucleotideIndex;
	Coordinate kmerIndex;
	KmerOccurence *kmerOccurence;
	int wordLength = getWordLength(graph);

	PassageMarkerI marker = NULL_IDX;
	PassageMarkerI previousMarker = NULL_IDX;
	Node *node = NULL;
	Node *previousNode = NULL;
	Coordinate coord = 0;
	Coordinate previousCoord = 0;
	Nucleotide nucleotide;
	boolean reversed;

	IDnum refID;
	Coordinate refCoord = 0;
	ReferenceMapping * refMap;
	Annotation * annotation = annotations;
	Coordinate index = 0;
	Coordinate uniqueIndex = 0;
	Coordinate annotIndex = 0;
	IDnum annotCount = 0;
	SmallNodeList * nodePile = NULL;

	// Neglect any string shorter than WORDLENGTH :
	if (getLength(tString) < wordLength)
		return;

	clearKmer(&word);
	clearKmer(&antiWord);

	// Fill in the initial word : 
	for (readNucleotideIndex = 0;
	     readNucleotideIndex < wordLength - 1; readNucleotideIndex++) {
		nucleotide = getNucleotide(readNucleotideIndex, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}
	}

	// Go through sequence
	// printf("len %d\n", getLength(tString));
	while (readNucleotideIndex < getLength(tString)) {
		nucleotide = getNucleotide(readNucleotideIndex++, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}

		// Update annotation if necessary
		if (annotCount < annotationCount && annotIndex == getAnnotationLength(annotation)) {
			annotation = getNextAnnotation(annotation);
			annotCount++;
			annotIndex = 0;
		}

		// Search for reference mapping
		if (category == REFERENCE) {
			if (referenceMappings) 
				refMap = findReferenceMapping(seqID, index, referenceMappings, referenceMappingCount);
			else 
				refMap = NULL;

			if (refMap) {
				node = getNodeInGraph(graph, refMap->nodeID);
				if (refMap->nodeID > 0) {
					coord = refMap->nodeStart + (index - refMap->referenceStart);
				} else {
					coord = getNodeLength(node) - refMap->nodeStart - refMap->length + (index - refMap->referenceStart);
				}
			} else  {
				node = NULL;
			}
		}
		// Search for reference-based mapping
		else if (annotCount < annotationCount && uniqueIndex >= getPosition(annotation) && getAnnotSequenceID(annotation) <= refCount && getAnnotSequenceID(annotation) >= -refCount) {
			refID = getAnnotSequenceID(annotation);
			if (refID > 0)
				refCoord = getStart(annotation) + annotIndex; 
			else
				refCoord = getStart(annotation) - annotIndex; 
			
			refMap = findReferenceMapping(refID, refCoord, referenceMappings, referenceMappingCount);
			// If success
			if (refMap) {
				if (refID > 0) {
					node = getNodeInGraph(graph, refMap->nodeID);
					if (refMap->nodeID > 0) {
						coord = refMap->nodeStart + (refCoord - refMap->referenceStart);
					} else {
						coord = getNodeLength(node) - refMap->nodeStart - refMap->length + (refCoord - refMap->referenceStart);
					}
				} else {
					node = getNodeInGraph(graph, -refMap->nodeID);
					if (refMap->nodeID > 0) {
						coord =  getNodeLength(node) - refMap->nodeStart - (refCoord - refMap->referenceStart) - 1;
					} else {
						coord = refMap->nodeStart + refMap->length - (refCoord - refMap->referenceStart) - 1;
					}
				}
			} else  {
				node = NULL;
				if (previousNode)
					break;
			}
		}		
		// Search in table
		else {
			reversed = false;
			if (double_strand) {
				if (compareKmers(&word, &antiWord) <= 0) {
					kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
								       kmerTable);
				} else { 
					kmerOccurence =
					       findKmerInKmerOccurenceTable(&antiWord,
						kmerTable);
					reversed = true;
				}
			} else {
				if (!second_in_pair) {
					kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
								       kmerTable);
				} else { 
					kmerOccurence =
					       findKmerInKmerOccurenceTable(&antiWord,
						kmerTable);
					reversed = true;
				}
			}
			
			if (kmerOccurence) {
				if (!reversed) {
					node = getNodeInGraph(graph, getKmerOccurenceNodeID(kmerOccurence));
					coord = getKmerOccurencePosition(kmerOccurence);
				} else {
					node = getNodeInGraph(graph, -getKmerOccurenceNodeID(kmerOccurence));
					coord = getNodeLength(node) - getKmerOccurencePosition(kmerOccurence) - 1;
				}
			} else {
				node = NULL;
				if (previousNode) 
					break;
			}
                        //printf(" T:%d  node:%d \n", omp_get_thread_num(), node);
		}

		// Increment positions
		if (annotCount < annotationCount && uniqueIndex >= getPosition(annotation)) 
			annotIndex++;
		else
			uniqueIndex++;

		// Fill in graph
		if (node)
		{
#ifdef _OPENMP
			lockNode(node);
#endif
			kmerIndex = readNucleotideIndex - wordLength;

			if (previousNode == node
			    && previousCoord == coord - 1) {
				if (category / 2 >= CATEGORIES) {
					setPassageMarkerFinish(marker,
							       kmerIndex +
							       1);
					setFinishOffset(marker,
							getNodeLength(node)
							- coord - 1);
				} else {
#ifndef SINGLE_COV_CAT
					incrementVirtualCoverage(node, category / 2, 1);
					incrementOriginalVirtualCoverage(node, category / 2, 1);
#else
					incrementVirtualCoverage(node, 1);
#endif
				}
#ifdef _OPENMP
				unLockNode(node);
#endif
			} else {
				if (category / 2 >= CATEGORIES) {
					marker =
					    newPassageMarker(seqID,
							     kmerIndex,
							     kmerIndex + 1,
							     coord,
							     getNodeLength
							     (node) -
							     coord - 1);
					transposePassageMarker(marker,
							       node);
					connectPassageMarkers
					    (previousMarker, marker,
					     graph);
					previousMarker = marker;
				} else {
					if (readTracking) {
						if (!isNodeMemorized(node, nodePile)) {
							addReadStart(node,
								     seqID,
								     coord,
								     graph,
								     kmerIndex);
							memorizeNode(node, &nodePile);
						} else {
							blurLastShortReadMarker
							    (node, graph);
						}
					}

#ifndef SINGLE_COV_CAT
					incrementVirtualCoverage(node, category / 2, 1);
					incrementOriginalVirtualCoverage(node, category / 2, 1);
#else
					incrementVirtualCoverage(node, 1);
#endif
				}
#ifdef _OPENMP
				lockTwoNodes(node, previousNode);
#endif

#pragma omp critical (xx3)  // pkr test
{
				if (category != REFERENCE)
					createArc(previousNode, node, graph);
}


#ifdef _OPENMP
				unLockTwoNodes(node, previousNode);
#endif
			}

			previousNode = node;
			previousCoord = coord;
		}
		index++;
	}
	// printKmer(&word);

	if (readTracking && category / 2 < CATEGORIES)
		unMemorizeNodes(&nodePile);
}

static void fillUpGraph(ReadSet * reads,
			KmerOccurenceTable * kmerTable,
			Graph * graph,
			boolean readTracking,
			boolean double_strand,
			ReferenceMapping * referenceMappings,
			Coordinate referenceMappingCount,
			IDnum refCount,
			char * roadmapFilename)
{
	IDnum readIndex;
	RoadMapArray *roadmap = NULL;
	Coordinate *annotationOffset = NULL;
	struct timeval start, end, diff;
	
	if (referenceMappings)
	{
		roadmap = importRoadMapArray(roadmapFilename);
		annotationOffset = callocOrExit(reads->readCount, Coordinate);
		for (readIndex = 1; readIndex < reads->readCount; readIndex++)
			annotationOffset[readIndex] = annotationOffset[readIndex - 1]
						      + getAnnotationCount(getRoadMapInArray(roadmap, readIndex - 1));
	}

	resetNodeStatus(graph);
	// Allocate memory for the read pairs
	if (!readStartsAreActivated(graph))
		activateReadStarts(graph);

	gettimeofday(&start, NULL);
#ifdef _OPENMP
	initSmallNodeListMemory();
	createNodeLocks(graph);
	#pragma omp parallel for
#endif
	for (readIndex = refCount; readIndex < reads->readCount; readIndex++)
	{
		Annotation * annotations = NULL;
		IDnum annotationCount = 0;
		Category category;
		boolean second_in_pair;

		if (readIndex % 1000000 == 0)
			velvetLog("Ghost Threading through reads %ld / %ld\n",
				  (long) readIndex, (long) reads->readCount);

		category = reads->categories[readIndex];
		second_in_pair = reads->categories[readIndex] & 1 && isSecondInPair(reads, readIndex);

		if (referenceMappings)
		{
			annotationCount = getAnnotationCount(getRoadMapInArray(roadmap, readIndex));
			annotations = getAnnotationInArray(roadmap->annotations, annotationOffset[readIndex]);
		}
	
		ghostThreadSequenceThroughGraph(getTightStringInArray(reads->tSequences, readIndex),
						kmerTable,
						graph, readIndex + 1,
						category,
						readTracking, double_strand,
						referenceMappings, referenceMappingCount,
					  	refCount, annotations, annotationCount,
						second_in_pair);
	}
	createNodeReadStartArrays(graph);
	gettimeofday(&end, NULL);
	timersub(&end, &start, &diff);
	velvetLog(" === Ghost-Threaded in %ld.%06ld s\n", (long) diff.tv_sec, (long) diff.tv_usec);

	gettimeofday(&start, NULL);
#ifdef _OPENMP
	int threads = omp_get_max_threads();
	if (threads > 32)
		threads = 32;

	#pragma omp parallel for num_threads(threads)
#endif
	for (readIndex = 0; readIndex < reads->readCount; readIndex++)
	{
		Annotation * annotations = NULL;
		IDnum annotationCount = 0;
		Category category;
		boolean second_in_pair;

		if (readIndex % 1000000 == 0)
			velvetLog("Threading through reads %li / %li\n",
				  (long) readIndex, (long) reads->readCount);

		category = reads->categories[readIndex];
		second_in_pair = reads->categories[readIndex] % 2 && isSecondInPair(reads, readIndex);

		if (referenceMappings)
		{
			annotationCount = getAnnotationCount(getRoadMapInArray(roadmap, readIndex));
			annotations = getAnnotationInArray(roadmap->annotations, annotationOffset[readIndex]);
		}

		threadSequenceThroughGraph(getTightStringInArray(reads->tSequences, readIndex),
					   kmerTable,
					   graph, readIndex + 1, category,
					   readTracking, double_strand,
					   referenceMappings, referenceMappingCount,
					   refCount, annotations, annotationCount, second_in_pair);
	}
	gettimeofday(&end, NULL);
	timersub(&end, &start, &diff);
	velvetLog(" === Threaded in %ld.%06ld s\n", (long) diff.tv_sec, (long) diff.tv_usec);

#ifdef _OPENMP
	free(nodeLocks);
	nodeLocks = NULL;
#endif

	if (referenceMappings)
	{
		destroyRoadMapArray(roadmap);
		free (annotationOffset);
	}

	orderNodeReadStartArrays(graph);

	destroySmallNodeListMemmory();

	destroyKmerOccurenceTable(kmerTable);
}

Graph *importPreGraph(char *preGraphFilename, ReadSet * reads, char * roadmapFilename, 
		      boolean readTracking, short int accelerationBits)
{
	boolean double_strand = false;
	Graph *graph = readPreGraphFile(preGraphFilename, &double_strand);
	Coordinate referenceMappingCount = 0;
	IDnum referenceCount = 0;

	if (nodeCount(graph) == 0)
		return graph;

	// If necessary compile reference -> node
	ReferenceMapping * referenceMappings = computeReferenceMappings(preGraphFilename, reads, &referenceMappingCount, &referenceCount); 
	// Node -> reference maps
	NodeMask * nodeMasks = computeNodeMasks(referenceMappings, referenceMappingCount, graph);

	// Map k-mers to nodes
	KmerOccurenceTable *kmerTable =
	    referenceGraphKmers(preGraphFilename, accelerationBits, graph, double_strand, nodeMasks, referenceMappingCount);

	free(nodeMasks);

	// Map sequences -> kmers -> nodes
	fillUpGraph(reads, kmerTable, graph, readTracking, double_strand, referenceMappings, referenceMappingCount, referenceCount, roadmapFilename);

	free(referenceMappings);

	return graph;
}

static void addReadsToGraph(TightString * tString,
				       KmerOccurenceTable * kmerTable,
				       Graph * graph,
				       IDnum seqID, Category category,
				       boolean readTracking,
				       boolean double_strand,
				       boolean second_in_pair)
{
	Kmer word;
	Kmer antiWord;
	Coordinate readNucleotideIndex;
	Coordinate kmerIndex;
	KmerOccurence *kmerOccurence;
	int wordLength = getWordLength(graph);

	Node *node = NULL;
	Node *previousNode = NULL;
	Coordinate coord = 0;
	Coordinate previousCoord = 0;
	Nucleotide nucleotide;
	boolean reversed;

	Coordinate index = 0;
	SmallNodeList * nodePile = NULL;

	// Neglect any read which will not be short paired
	if (category / 2 >= CATEGORIES)
		return;
		
	// Neglect any string shorter than WORDLENGTH :
	if (getLength(tString) < wordLength)
		return;

	clearKmer(&word);
	clearKmer(&antiWord);

	// Fill in the initial word : 
	for (readNucleotideIndex = 0;
	     readNucleotideIndex < wordLength - 1; readNucleotideIndex++) {
		nucleotide = getNucleotide(readNucleotideIndex, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}
	}

	// Go through sequence
	// printf("len %d\n", getLength(tString));
	while (readNucleotideIndex < getLength(tString)) {
		nucleotide = getNucleotide(readNucleotideIndex++, tString);
		pushNucleotide(&word, nucleotide);
		if (double_strand || second_in_pair) {
#ifdef COLOR
			reversePushNucleotide(&antiWord, nucleotide);
#else
			reversePushNucleotide(&antiWord, 3 - nucleotide);
#endif
		}

		// Search in table
		reversed = false;
		if (double_strand) {
			if (compareKmers(&word, &antiWord) <= 0) {
				kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
							kmerTable);
			} else { 
				kmerOccurence =
					findKmerInKmerOccurenceTable(&antiWord,
							kmerTable);
				reversed = true;
			}
		} else {
			if (!second_in_pair) {
				kmerOccurence =
					findKmerInKmerOccurenceTable(&word,
							kmerTable);
			} else { 
				kmerOccurence =
					findKmerInKmerOccurenceTable(&antiWord,
							kmerTable);
				reversed = true;
			}
		}

		if (kmerOccurence) {
			if (!reversed) {
				node = getNodeInGraph(graph, getKmerOccurenceNodeID(kmerOccurence));
				coord = getKmerOccurencePosition(kmerOccurence);
			} else {
				node = getNodeInGraph(graph, -getKmerOccurenceNodeID(kmerOccurence));
				coord = getNodeLength(node) - getKmerOccurencePosition(kmerOccurence) - 1;
			}
		} else {
			node = NULL;
			if (previousNode) 
				break;
		}

		// Fill in graph
		if (node)
		{
#ifdef _OPENMP
			lockNode(node);
#endif
			kmerIndex = readNucleotideIndex - wordLength;

			if (previousNode != node || previousCoord != coord -1) {
				if (!isNodeMemorized(node, nodePile)) {
					addReadStart(node,
							seqID,
							coord,
							graph,
							kmerIndex);
					memorizeNode(node, &nodePile);
				} else {
					blurLastShortReadMarker
						(node, graph);
				}
			}
#ifdef _OPENMP
			unLockNode(node);
#endif
			previousNode = node;
			previousCoord = coord;
		}
		index++;
	}
	// printKmer(&word);

	if (category / 2 < CATEGORIES)
		unMemorizeNodes(&nodePile);
}

static void fillUpConnectedGraph(ReadSet * reads,
			KmerOccurenceTable * kmerTable,
			Graph * graph,
			boolean readTracking,
			boolean double_strand)
{
	IDnum refCount = 0;   // refs not present in connected graphs
	IDnum readIndex;
	struct timeval start, end, diff;
	
	resetNodeStatus(graph);
	// Allocate memory for the read pairs
	if (!readStartsAreActivated(graph))
		activateReadStarts(graph);

	gettimeofday(&start, NULL);
#ifdef _OPENMP
	initSmallNodeListMemory();
	createNodeLocks(graph);
	#pragma omp parallel for
#endif
	for (readIndex = refCount; readIndex < reads->readCount; readIndex++)
	{
		Category category;
		boolean second_in_pair;

		if (readIndex % 1000000 == 0)
			velvetLog("Ghost Threading through reads %ld / %ld\n",
				  (long) readIndex, (long) reads->readCount);

		category = reads->categories[readIndex];
		second_in_pair = reads->categories[readIndex] & 1 && isSecondInPair(reads, readIndex);

		// referenceMappings = NULL, referenceMappingCount = 0
		// refCount = 0, annotations = NULL, annotationCount = 0
		ghostThreadSequenceThroughGraph(getTightStringInArray(reads->tSequences, readIndex),
						kmerTable,
						graph, readIndex + 1,
						category,
						readTracking, double_strand,
						NULL, 0,
					  	0, NULL, 0,
						second_in_pair);
	}
	createNodeReadStartArrays(graph);
	gettimeofday(&end, NULL);
	timersub(&end, &start, &diff);
	velvetLog(" === Ghost-Threaded in %ld.%06ld s\n", diff.tv_sec, diff.tv_usec);

	gettimeofday(&start, NULL);
#ifdef _OPENMP
	int threads = omp_get_max_threads();
	if (threads > 32)
		threads = 32;

	#pragma omp parallel for num_threads(threads)
#endif
	for (readIndex = 0; readIndex < reads->readCount; readIndex++)
	{
		Category category;
		boolean second_in_pair;

		if (readIndex % 1000000 == 0)
			velvetLog("Adding reads %li / %li\n",
				  (long) readIndex, (long) reads->readCount);

		category = reads->categories[readIndex];
		second_in_pair = reads->categories[readIndex] % 2 && isSecondInPair(reads, readIndex);

		addReadsToGraph(getTightStringInArray(reads->tSequences, readIndex),
					   kmerTable,
					   graph, readIndex + 1, category,
					   readTracking, double_strand, second_in_pair);
	}
	gettimeofday(&end, NULL);
	timersub(&end, &start, &diff);
	velvetLog(" === Threaded in %ld.%06ld s\n", diff.tv_sec, diff.tv_usec);

#ifdef _OPENMP
	free(nodeLocks);
	nodeLocks = NULL;
#endif

	orderNodeReadStartArrays(graph);

	destroySmallNodeListMemmory();

	destroyKmerOccurenceTable(kmerTable);
}

Graph *importConnectedGraph(char *connectedGraphFilename, ReadSet * reads, char * roadmapFilename,
		      boolean readTracking, short int accelerationBits)
{
	boolean double_strand = false;
	Graph *graph = readConnectedGraphFile(connectedGraphFilename, &double_strand);

	if (nodeCount(graph) == 0)
		return graph;

	if (readTracking) {
		Coordinate referenceMappingCount = 0;
		NodeMask * nodeMasks = NULL;

		// Map k-mers to nodes
		KmerOccurenceTable *kmerTable =
			referenceGraphKmers(connectedGraphFilename, accelerationBits, graph, doubleStrandedGraph(graph), nodeMasks, referenceMappingCount);

		// Map sequences -> kmers -> nodes
		fillUpConnectedGraph(reads, kmerTable, graph, readTracking, double_strand);
	}

	return graph;
}
