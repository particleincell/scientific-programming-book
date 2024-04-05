#include <stdio.h>

int main(int num_args, char* args[]) {
	printf("num_args is %d\n",num_args);
	for (int i=0; i<num_args; i++) 
		printf("%s\n",args[i]);
	return 0;
}
