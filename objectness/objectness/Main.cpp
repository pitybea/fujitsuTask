// Objectness.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include <iostream>
#include <intrin.h> 
#include <direct.h>
#include <Windows.h>

void gotoDir(const string& s)
{
	_chdir(s.c_str());
}

vector<string> readFileNames(const string& s)
{
	FILE* fp=fopen(s.c_str(),"r");
	int n;
	fscanf(fp,"%d\n",&n);
	vector<string> rslt;
	rslt.resize(n);
	for (int i = 0; i < n; i++)
	{
		char tem[35];
		fscanf(fp,"%s\n",&tem);
		rslt[i]=tem;
	}
	fclose(fp);
	return rslt;
}
void RunmyObjectness(CStr &resName, double base, int W, int NSS, int numPerSz);
void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz);

void illutrateLoG()
{
	for (float delta = 0.5f; delta < 1.1f; delta+=0.1f){
		Mat f = Objectness::aFilter(delta, 8);
		normalize(f, f, 0, 1, NORM_MINMAX);
		CmShow::showTinyMat(format("D=%g", delta), f);
	}
	waitKey(0);
}

int main(int argc, char* argv[])
{


	//CStr wkDir = "D:/WkDir/DetectionProposals/VOC2007/Local/";
	//illutrateLoG();
	/*FileStorage fs;
	fs.open("D:\\000012.yml", FileStorage::READ);
	auto fn=fs["annotation"]["object"];
	cout<<fn.isSeq();
	string strXmin, strYmin, strXmax, strYmax;
	fn["bndbox"]["xmin"] >> strXmin;
	fn["bndbox"]["ymin"] >> strYmin;
	fn["bndbox"]["xmax"] >> strXmax;
	fn["bndbox"]["ymax"] >> strYmax;*/
	RunmyObjectness("WinRecall.m", 2, 8, 2, 130);

	return 0;
}


void RunmyObjectness(CStr &resName, double base, int W, int NSS, int numPerSz)
{
	srand((unsigned int)time(NULL));
	DataSetVOC voc2007("../VOC2007/");
//	voc2007.cvt2OpenCVYml("D:\\project\\objectness\\VOC2007\\Annotations\\");
//	voc2007.loadAnnotations();
	//voc2007.loadDataGenericOverCls();

//	printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
//	printf("%s Base = %g, W = %d, NSS = %d, perSz = %d\n", _S(resName), base, W, NSS, numPerSz);
	
	Objectness objNess(voc2007, base, W, NSS);

	vector<vector<Vec4i>> boxesTests;

	gotoDir("D:\\ZPWang\\vanish\\dataset\\ny");
	auto sth=readFileNames("img.lst");
	objNess.boundboxesforImgs(sth,boxesTests,numPerSz);
	//objNess.getObjBndBoxesForTests(boxesTests, 250);
//	objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
	//objNess.getRandomBoxes(boxesTests);

	//objNess.evaluatePerClassRecall(boxesTests, resName, 1000);
	//objNess.illuTestReults(boxesTests);
}



void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz)
{
	srand((unsigned int)time(NULL));
	DataSetVOC voc2007("../VOC2007/");
//	voc2007.cvt2OpenCVYml("D:\\project\\objectness\\VOC2007\\Annotations\\");
	voc2007.loadAnnotations();
	//voc2007.loadDataGenericOverCls();

	printf("Dataset:`%s' with %d training and %d testing\n", _S(voc2007.wkDir), voc2007.trainNum, voc2007.testNum);
	printf("%s Base = %g, W = %d, NSS = %d, perSz = %d\n", _S(resName), base, W, NSS, numPerSz);
	
	Objectness objNess(voc2007, base, W, NSS);

	vector<vector<Vec4i>> boxesTests;
	//objNess.getObjBndBoxesForTests(boxesTests, 250);
	objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
	//objNess.getRandomBoxes(boxesTests);

	//objNess.evaluatePerClassRecall(boxesTests, resName, 1000);
	//objNess.illuTestReults(boxesTests);
}
