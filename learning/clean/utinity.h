#include <vector>
#include <string>
#include <windows.h>
#include <direct.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

#pragma once 


pair<vector<vector<double> >,vector<int> > parallelKMeans(const vector<vector<double> >& dataset,int kCenter=-1,int maxIterationNumber=1000);

pair<vector<vector<double> >,vector<int> > parallelKMeans2(const vector<vector<double> >& dataset,int kCenter=-1,int maxIterationNumber=1000);

template<class T>
void FromSmall(vector<T>& p,int n,vector<int>& index)
{
	int k,j,i;
	T t;
	int ii;
	k=n/2;
	while(k>0)
	{
		for(j=k;j<=n-1;j++)
		{
			t=p[j];  ii=index[j];  i=j-k;
			while((i>=0)&&(p[i]>t))
			{
				p[i+k]=p[i];  index[i+k]=index[i];  i=i-k;
			}
			p[i+k]=t;  index[i+k]=ii;
		}
		k=k/2;
	}
};



inline bool fileExists (const string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

double dis(const vector<double>& a,const vector<double>& b);
