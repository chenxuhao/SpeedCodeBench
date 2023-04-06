#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "ops.h"

int isInteger(char *str){
  if (*str == '\0'){
    return 0;
  }
  for(; *str != '\0'; str++){
    if (*str < 48 || *str > 57){	// digit characters (need to include . if checking for float)
      return 0;
    }
  }
  return 1;
}

//int order = DEFAULT_ORDER;
//bool verbose_output = false;

void kernel_cpu(int cores_arg,record *records, knode *knodes, long knodes_elem, int korder,
                long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans);

void kernel_cpu_2(int cores_arg, knode *knodes, long knodes_elem, int korder,
                  long maxheight, int count, long *currKnode, long *offset, long *lastKnode,
                  long *offset_2, int *start, int *end, int *recstart, int *reclength);

int main(int argc, char** argv ) {
  // assing default values
  int cur_arg;
  int cores_arg =1;
  char *input_file = NULL;
  char *command_file = NULL;
  char *output="output.txt";
  FILE * pFile;

  // go through arguments
  for(cur_arg=1; cur_arg<argc; cur_arg++){
    if(strcmp(argv[cur_arg], "cores")==0){
      // check if value provided
      if(argc>=cur_arg+1){
        // check if value is a number
        if(isInteger(argv[cur_arg+1])==1){
          cores_arg = atoi(argv[cur_arg+1]);
          if(cores_arg<0){
            printf("ERROR: Wrong value to cores parameter, cannot be <=0\n");
            return -1;
          }
          cur_arg = cur_arg+1;
        }
        // value is not a number
        else{
          printf("ERROR: Value to cores parameter in not a number\n");
          return 0;
        }
      }
    }
    // check if -file
    else if(strcmp(argv[cur_arg], "file")==0){
      // check if value provided
      if(argc>=cur_arg+1){
        input_file = argv[cur_arg+1];
        cur_arg = cur_arg+1;
        // value is not a number
      }
      // value not provided
      else{
        printf("ERROR: Missing value to -file parameter\n");
        return -1;
      }
    }
    else if(strcmp(argv[cur_arg], "command")==0){
      // check if value provided
      if(argc>=cur_arg+1){
        command_file = argv[cur_arg+1];
        cur_arg = cur_arg+1;
        // value is not a number
      }
      // value not provided
      else{
        printf("ERROR: Missing value to command parameter\n");
        return -1;
      }
    }
  }
  // Print configuration
  if((input_file==NULL)||(command_file==NULL))
    printf("Usage: ./b+tree file input_file command command_list\n");

  // For debug
  printf("Input File: %s \n", input_file);
  printf("Command File: %s \n", command_file);

  FILE * commandFile;
  long lSize;
  char * commandBuffer;
  size_t result;

  commandFile = fopen ( command_file, "rb" );
  if (commandFile==NULL) {fputs ("Command File error",stderr); exit (1);}

  // obtain file size:
  fseek (commandFile , 0 , SEEK_END);
  lSize = ftell (commandFile);
  rewind (commandFile);

  // allocate memory to contain the whole file:
  commandBuffer = (char*) malloc (sizeof(char)*lSize);
  if (commandBuffer == NULL) {fputs ("Command Buffer memory error",stderr); exit (2);}

  // copy the file into the buffer:
  result = fread (commandBuffer,1,lSize,commandFile);
  if (result != lSize) {fputs ("Command file reading error",stderr); exit (3);}

  /* the whole file is now loaded in the memory buffer. */

  // terminate
  fclose (commandFile);

  // For Debug
  char *sPointer=commandBuffer;
  printf("Command Buffer: \n");
  printf("%s",commandBuffer);
  //


  pFile = fopen (output,"w+");
  if (pFile==NULL) 
    fputs ("Fail to open %s !\n",output);
  fprintf(pFile,"******starting******\n");
  fclose(pFile);


  // ------------------------------------------------------------60
  // general variables
  // ------------------------------------------------------------60

  FILE *file_pointer;
  node *root;
  root = NULL;
  record *r;
  int input;
  char instruction;
  order = DEFAULT_ORDER;
  verbose_output = false;

  //usage_1();  
  //usage_2();

  // ------------------------------------------------------------60
  // get input from file, if file provided
  // ------------------------------------------------------------60

  if (input_file != NULL) {

    printf("Getting input from file %s...\n", argv[1]);

    // open input file
    file_pointer = fopen(input_file, "r");
    if (file_pointer == NULL) {
      perror("Failure to open input file.");
      exit(EXIT_FAILURE);
    }

    // get # of numbers in the file
    fscanf(file_pointer, "%d\n", &input);
    size = input;

    // save all numbers
    while (!feof(file_pointer)) {
      fscanf(file_pointer, "%d\n", &input);
      root = insert(root, input, input);
    }

    // close file
    fclose(file_pointer);
    //print_tree(root);
    //printf("Height of tree = %d\n", height(root));

  }
  else{
    printf("ERROR: Argument -file missing\n");
    return 0;
  }

  // ------------------------------------------------------------60
  // get tree statistics
  // ------------------------------------------------------------60

  printf("Transforming data to a GPU suitable structure...\n");
  long mem_used = transform_to_cuda(root,0);
  maxheight = height(root);
  long rootLoc = (long)knodes - (long)mem;

  // ------------------------------------------------------------60
  // process commands
  // ------------------------------------------------------------60
  char *commandPointer=commandBuffer;

  printf("Waiting for command\n");
  printf("> ");
  while (sscanf(commandPointer, "%c", &instruction) != EOF) {
    commandPointer++;
    switch (instruction) {
      // ----------------------------------------40
      // Insert
      // ----------------------------------------40

      case 'i':
        {
          scanf("%d", &input);
          while (getchar() != (int)'\n');
          root = insert(root, input, input);
          print_tree(root);
          break;
        }

        // ----------------------------------------40
        // n/a
        // ----------------------------------------40

      case 'f':
        {
        }

        // ----------------------------------------40
        // find
        // ----------------------------------------40

      case 'p':
        {
          scanf("%d", &input);
          while (getchar() != (int)'\n');
          r = find(root, input, instruction == 'p');
          if (r == NULL)
            printf("Record not found under key %d.\n", input);
          else 
            printf("Record found: %d\n",r->value);
          break;
        }

        // ----------------------------------------40
        // delete value
        // ----------------------------------------40

      case 'd':
        {
          scanf("%d", &input);
          while (getchar() != (int)'\n');
          root = (node *) deleteVal(root, input);
          print_tree(root);
          break;
        }

        // ----------------------------------------40
        // destroy tree
        // ----------------------------------------40

      case 'x':
        {
          while (getchar() != (int)'\n');
          root = destroy_tree(root);
          print_tree(root);
          break;
        }

        // ----------------------------------------40
        // print leaves
        // ----------------------------------------40

      case 'l':
        {
          while (getchar() != (int)'\n');
          print_leaves(root);
          break;
        }

        // ----------------------------------------40
        // print tree
        // ----------------------------------------40

      case 't':
        {
          while (getchar() != (int)'\n');
          print_tree(root);
          break;
        }

        // ----------------------------------------40
        // toggle verbose output
        // ----------------------------------------40

      case 'v':
        {
          while (getchar() != (int)'\n');
          verbose_output = !verbose_output;
          break;
        }

        // ----------------------------------------40
        // quit
        // ----------------------------------------40

      case 'q':
        {
          while (getchar() != (int)'\n');
          return EXIT_SUCCESS;
        }

        // ----------------------------------------40
        // [OpenMP] find K (initK, findK)
        // ----------------------------------------40

      case 'k':
        {

          // get # of queries from user
          int count;
          sscanf(commandPointer, "%d", &count);
          while(*commandPointer!=32 && commandPointer!='\n')
            commandPointer++;

          printf("\n ******command: k count=%d \n",count);

          if(count > 65535){
            printf("ERROR: Number of requested querries should be 65,535 at most. (limited by # of CUDA blocks)\n");
            exit(0);
          }

          // INPUT: records CPU allocation (setting pointer in mem variable)
          record *records = (record *)mem;
          long records_elem = (long)rootLoc / sizeof(record);
          long records_mem = (long)rootLoc;
          // printf("records_elem=%d, records_unit_mem=%d, records_mem=%d\n", (int)records_elem, sizeof(record), (int)records_mem);

          // INPUT: knodes CPU allocation (setting pointer in mem variable)
          knode *knodes = (knode *)((long)mem + (long)rootLoc);
          long knodes_elem = ((long)(mem_used) - (long)rootLoc) / sizeof(knode);
          long knodes_mem = (long)(mem_used) - (long)rootLoc;
          // printf("knodes_elem=%d, knodes_unit_mem=%d, knodes_mem=%d\n", (int)knodes_elem, sizeof(knode), (int)knodes_mem);

          // INPUT: currKnode CPU allocation
          long *currKnode;
          currKnode = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset(currKnode, 0, count*sizeof(long));

          // INPUT: offset CPU allocation
          long *offset;
          offset = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset(offset, 0, count*sizeof(long));

          // INPUT: keys CPU allocation
          int *keys;
          keys = (int *)malloc(count*sizeof(int));
          // INPUT: keys CPU initialization
          int i;
          for(i = 0; i < count; i++){
            keys[i] = (rand()/(float)RAND_MAX)*size;
          }

          // OUTPUT: ans CPU allocation
          record *ans = (record *)malloc(sizeof(record)*count);
          // OUTPUT: ans CPU initialization
          for(i = 0; i < count; i++){
            ans[i].value = -1;
          }

          // New OpenMP kernel, same algorighm across all versions(OpenMP, CUDA, OpenCL) for comparison purposes
          kernel_cpu(	cores_arg,
              records,
              knodes,
              knodes_elem,
              order,
              maxheight,
              count,
              currKnode,
              offset,
              keys,
              ans);

          // Original OpenMP kernel, different algorithm
          // int j;
          // for(j = 0; j < count; j++){
          // find(	root,				// node *

          // keys[j],			// int
          // false);				// bool
          // }


          pFile = fopen (output,"aw+");
          if (pFile==NULL)
          {
            fputs ("Fail to open %s !\n",output);
          }

          fprintf(pFile,"\n ******command: k count=%d \n",count);
          for(i = 0; i < count; i++){
            fprintf(pFile, "%d    %d\n",i, ans[i].value);
          }
          fprintf(pFile, " \n");
          fclose(pFile);

          // free memory
          free(currKnode);
          free(offset);
          free(keys);
          free(ans);

          // break out of case
          break;

        }

        // ----------------------------------------40
        // find range
        // ----------------------------------------40

      case 'r':
        {
          int start, end;
          scanf("%d", &start);
          scanf("%d", &end);
          if(start > end){
            input = start;
            start = end;
            end = input;
          }
          printf("For range %d to %d, ",start,end);
          list_t * ansList;
          ansList = findRange(root, start, end);
          printf("%d records found\n", list_get_length(ansList));
          //list_iterator_t iter;
          free(ansList);
          break;
        }

        // ----------------------------------------40
        // [OpenMP] find Range K (initK, findRangeK)
        // ----------------------------------------40

      case 'j':
        {

          // get # of queries from user
          int count;
          sscanf(commandPointer, "%d", &count);
          while(*commandPointer!=32 && commandPointer!='\n')
            commandPointer++;

          int rSize;
          sscanf(commandPointer, "%d", &rSize);
          while(*commandPointer!=32 && commandPointer!='\n')
            commandPointer++;

          printf("\n******command: j count=%d, rSize=%d \n",count, rSize);

          if(rSize > size || rSize < 0) {
            printf("Search range size is larger than data set size %d.\n", (int)size);
            exit(0);
          }

          // INPUT: knodes CPU allocation (setting pointer in mem variable)
          knode *knodes = (knode *)((long)mem + (long)rootLoc);
          long knodes_elem = ((long)(mem_used) - (long)rootLoc) / sizeof(knode);
          long knodes_mem = (long)(mem_used) - (long)rootLoc;
          // printf("knodes_elem=%d, knodes_unit_mem=%d, knodes_mem=%d\n", (int)knodes_elem, sizeof(knode), (int)knodes_mem);

          // INPUT: currKnode CPU allocation
          long *currKnode;
          currKnode = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset (currKnode, 0, count*sizeof(long));

          // INPUT: offset CPU allocation
          long *offset;
          offset = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset (offset, 0, count*sizeof(long));

          // INPUT: lastKnode CPU allocation
          long *lastKnode;
          lastKnode = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset (lastKnode, 0, count*sizeof(long));

          // INPUT: offset_2 CPU allocation
          long *offset_2;
          offset_2 = (long *)malloc(count*sizeof(long));
          // INPUT: offset CPU initialization
          memset (offset_2, 0, count*sizeof(long));

          // INPUT: start, end CPU allocation
          int *start;
          start = (int *)malloc(count*sizeof(int));
          int *end;
          end = (int *)malloc(count*sizeof(int));
          // INPUT: start, end CPU initialization
          int i;
          for(i = 0; i < count; i++){
            start[i] = (rand()/(float)RAND_MAX)*size;
            end[i] = start[i]+rSize;
            if(end[i] >= size){ 
              start[i] = start[i] - (end[i] - size);
              end[i]= size-1;
            }
          }

          // INPUT: recstart, reclenght CPU allocation
          int *recstart;
          recstart = (int *)malloc(count*sizeof(int));
          int *reclength;
          reclength = (int *)malloc(count*sizeof(int));
          // OUTPUT: ans CPU initialization
          for(i = 0; i < count; i++){
            recstart[i] = 0;
            reclength[i] = 0;
          }

          // New kernel, same algorighm across all versions(OpenMP, CUDA, OpenCL) for comparison purposes
          kernel_cpu_2(cores_arg,
              knodes,
              knodes_elem,
              order,
              maxheight,
              count,
              currKnode,
              offset,
              lastKnode,
              offset_2,
              start,
              end,
              recstart,
              reclength);

          // Original [CPU] kernel, different algorithm
          // int k;
          // for(k = 0; k < count; k++){
          // findRange(	root,

          // start[k], 
          // end[k]);
          // }
          pFile = fopen (output,"aw+");
          if (pFile==NULL)
          {
            fputs ("Fail to open %s !\n",output);
          }

          fprintf(pFile,"\n******command: j count=%d, rSize=%d \n",count, rSize);				
          for(i = 0; i < count; i++){
            fprintf(pFile, "%d    %d    %d\n",i, recstart[i],reclength[i]);
          }
          fprintf(pFile, " \n");
          fclose(pFile);

          // free memory
          free(currKnode);
          free(offset);
          free(lastKnode);
          free(offset_2);
          free(start);
          free(end);
          free(recstart);
          free(reclength);

          // break out of case
          break;

        }
      default:
        {
          //usage_2();
          break;
        }
    }
    printf("> ");
  }
  printf("\n");
  free(mem);
  return EXIT_SUCCESS;
}

