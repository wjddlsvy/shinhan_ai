#include <stdio.h>
#include"MLP.h"

CMLP MultiLayer;

int main(void) {
	int hlayer[2] = { 3, 2 };

	MultiLayer.Create(2, 1, 2, hlayer);

	MultiLayer.LoadWeight((char*)"weight.txt");

	printf("******** 학습종료 ****************\n");

	MultiLayer.SaveWeight((char*)"weight.txt");
	return 0;
}