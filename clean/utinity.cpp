#include "utinity.h"

#include <time.h>

#include <assert.h>
#include "settings.inl"

using namespace FujitsuTask;

vector<int> shuffledOrder(int n)
{
	vector<int> result(n);
	vector<int> index(n);
	for(int i=0;i<n;++i)
	{
		index[i]=i;
	}

	for(int i=0;i<n;++i)
	{
		int tem=rand()%(n-i);
		result[i]=index[tem];
		index[tem]=index[n-i-1];

	}
	return result;
}

pair<vector<vector<double> >,vector<int> > parallelKMeans(const vector<vector<double> >& dataset,int kCenter)
{
	if(kCenter==-1)
		kCenter=dataset.size()/1000;

	if(kCenter<1) kCenter=1;

	assert(dataset.size()>=kCenter);
	vector<vector<double> > centers(kCenter,dataset[0]);
	vector<int> labels(dataset.size(),0);

	vector<int> initialCenterIndex=shuffledOrder(dataset.size());

	for(int i=0;i<centers.size();++i)
	{
		centers[i]=dataset[initialCenterIndex[i]];
	}
	
	pair<vector<vector<double> >,vector<int> > result;
	
	result.first=centers;
	result.second=labels;

	return result;

}