#pragma once
class CMLP {
public:
	CMLP();
	~CMLP();
	//신경망 구조선언을 위한 변수
	int m_iNumlnNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer; //hidden only
	int m_iNumTotalLayer; 
	int* m_NumNodes;

	double*** m_Weight;
	double** m_NodeOut;

	double* pInValue, * pOutValue;
	double* pCorrectOutValue;

	double** m_ErrorGradient;

	bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void Initw();
	double ActivationFunc(double weightsum);
public:
	void Forward();
	void BackPopagationLearning();
	bool SaveWeight(char* fname);
	bool LoadWeight(char* fname);
};