#include "utinity.h"

#include <time.h>


#include "settings.inl"

using namespace FujitsuTask;



double dis(const vector<double>& a,const vector<double>& b)
{
	assert(a.size()==b.size());

	double sum=0;
	for(int i=0;i<a.size();++i)
	{
		double tem=a[i]-b[i];
		sum+=tem*tem;
	}
	return sum;
}

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

vector<int> shuffledOrder(int n,int m)
{
	assert(n>=m);
	vector<int> result(m);
	vector<int> index(n);
	for(int i=0;i<n;++i)
	{
		index[i]=i;
	}

	for(int i=0;i<m;++i)
	{
		int tem=rand()%(n-i);
		result[i]=index[tem];
		index[tem]=index[n-i-1];

	}
	return result;
}

bool sameClusters(unordered_map<int, vector<int> >& a,const vector<int>& b)
{
	for(unordered_map<int, vector<int> >::iterator iter=a.begin();iter!=a.end();++iter)
	{
		int clusterIndex=b[iter->second[0]];


		for(int j=1;j<iter->second.size();++j)
		{
			if(b[iter->second[j]]!=clusterIndex)
				return false;
		}
	}

	return true;
}

int indexOfMostSimilar(const vector<double>& a,const vector<vector<double>>& dataset)
{
	int result=0;
	double minDis=dis(a,dataset[0]);

	for(int i=1;i<dataset.size();++i)
	{
		double tdis=dis(a,dataset[i]);
		if(tdis<minDis)
		{
			minDis=tdis;
			result=i;
		}
	}

	return result;
}

vector<double> calculateCenter(const vector<vector<double> >& dataset,const vector<int>& index)
{
	assert(index.size()>0);
	vector<double> center(dataset[0].size(),0.0);

	for(int i=0;i<index.size();++i)
		for(int j=0;j<center.size();++j)
			center[j]+=dataset[index[i]][j];

	for(int j=0;j<center.size();++j)
		center[j]/=index.size();

	return center;
}

pair<vector<vector<double> >,vector<int> > parallelKMeans(const vector<vector<double> >& dataset,int kCenter,int maxIterationNumber)
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
		//centers[i]=dataset[initialCenterIndex[i]];
		centers[i]=dataset[i];
	}
	
	for(int iterNumber=0;iterNumber<maxIterationNumber;++iterNumber)
	{
		cout<<iterNumber<<"th iteration\n "<<endl;
		vector<int> temLabels=labels;
		unordered_map<int,vector<int> > records;
		#pragma omp parallel for
		for(int i=0;i<parallelNumber;++i)
		{
			for(int j=i;j<dataset.size();j+=parallelNumber)
			{
				labels[j]=indexOfMostSimilar(dataset[j],centers);
			}
		}

		for(int i=0;i<labels.size();++i)
			records[labels[i]].push_back(i);

		int index=0;

		vector<vector<int> > centerIndex;
		
		for(unordered_map<int,vector<int>>::iterator iter=records.begin();iter!=records.end();++iter,++index)
		{
			vector<int>& oneCluster=iter->second;

			centerIndex.push_back(oneCluster);

			if(records.size()<centers.size())
			for(int j=0;j<oneCluster.size();++j)
			{
				labels[oneCluster[j]]=index;
			}
		}

		centers.resize(centerIndex.size(),dataset[0]);

		#pragma omp parallel for
		for(int i=0;i<parallelNumber;++i)
		{
			for(int j=i;j<centerIndex.size();j+=parallelNumber)
			{
				centers[j]=calculateCenter(dataset,centerIndex[j]);		
			}
		}




		if(iterNumber>0 && sameClusters(records,temLabels))
			break;
	}


	pair<vector<vector<double> >,vector<int> > result;
	
	result.first=centers;
	result.second=labels;

	return result;

}


int indexOfMostSimilar(const vector<double>& a,const vector<vector<double>>& dataset,const vector<bool>& dataFlag )
{
	assert(dataset.size()==dataFlag.size());
	int firstindex=0;
	while(firstindex<dataFlag.size()&& ( !dataFlag[firstindex] ))
		++firstindex;
	assert(firstindex<dataFlag.size());

	int result=firstindex;
	double minDis=dis(a,dataset[firstindex]);

	for(int i=firstindex;i<dataset.size();++i)
	{
		if(dataFlag[i])
		{
			double tdis=dis(a,dataset[i]);
			if(tdis<minDis)
			{
				minDis=tdis;
				result=i;
			}
		}
	}

	return result;
}


pair<vector<vector<double> >,vector<int> > parallelKMeans2(const vector<vector<double> >& dataset,int kCenter,int maxIterationNumber)
{
	if(kCenter==-1)
		kCenter=dataset.size()/1000;

	if(kCenter<1) kCenter=1;



	assert(dataset.size()>=kCenter);

	int dimension=dataset[0].size();

	vector<vector<double> > centers(kCenter,vector<double>(dataset[0].size(),0.0));
	vector<bool> goodCenterFlag(kCenter,true);

	vector<int> labels(dataset.size(),0);

	vector<vector<int> > paraClusterCount(parallelNumber,vector<int>(kCenter,0));
	vector<int> clusterCount(kCenter,0);

	vector<vector<vector<double> > > paraCenters(parallelNumber,vector<vector<double> >(kCenter,vector<double>(dataset[0].size(),0.0)));

//	vector<bool> centerChangeFlag(dataset.size(),false); 
	vector<bool> paraCenterChangeFlag(parallelNumber,false);

	vector<int> initialCenterIndex=shuffledOrder(dataset.size(),kCenter);

	for(int i=0;i<centers.size();++i)
	{
		//centers[i]=dataset[initialCenterIndex[i]];
		centers[i]=dataset[i];
	}

	for (int iterNumber = 0; iterNumber < maxIterationNumber; ++iterNumber)
	{
		
		#pragma omp parallel for
		for (int i = 0; i < parallelNumber; ++i)
		{
			for(int j=i;j<dataset.size();j+=parallelNumber)
			{
				int tlabel=indexOfMostSimilar(dataset[j],centers,goodCenterFlag);
				if(tlabel!=labels[j])
					paraCenterChangeFlag[i]=true;
				labels[j]=tlabel;
				++paraClusterCount[i][tlabel];
				for (int k = 0; k < dimension; k++)
				{
					paraCenters[i][tlabel][k]+=dataset[j][k];
				}
			}
		}
		#pragma omp parallel for
		for(int i=0;i<parallelNumber;++i)
		{
			for (int j = i; j < kCenter; j+=parallelNumber)
			{
				centers[j].resize(dimension,0.0);
				clusterCount[j]=0;
				for (int k = 0; k < paraCenters.size(); k++)
				{
					for (int l = 0; l < dimension; l++)
					{
						centers[j][l]+=paraCenters[k][j][l];
						paraCenters[k][j][l]=0.0;
					}
					clusterCount[j]+=paraClusterCount[k][j];
					paraClusterCount[k][j]=0;
				}
				if(clusterCount[j]==0)
					goodCenterFlag[j]=false;
				else
					for (int k = 0; k < dimension; k++)
					{
						centers[j][k]/=clusterCount[j];
					}
			}
		}
		bool noChange=true;
		for (int i = 0; i < parallelNumber; i++)
		{
			if(paraCenterChangeFlag[i]==true)
			{
				noChange=false;
				paraCenterChangeFlag[i]=false;
			}
			
		}

		cout<<"finished iteration NO. "<<iterNumber<<endl;
		if(noChange)
			break;

	}

	vector<vector<double> > trueCenters;

	unordered_map<int,int> minusNumber;
	int numEmptyCluster=0;

	for (int i = 0; i < goodCenterFlag.size(); i++)
	{
		if(goodCenterFlag[i])
		{
			trueCenters.push_back(centers[i]);
		}
		else
		{
			++numEmptyCluster;
		}
		if(numEmptyCluster>0)
			minusNumber[i]=numEmptyCluster;
	}
	if(numEmptyCluster>0)
	{
		#pragma omp parallel for
		for (int i = 0; i < parallelNumber; i++)
		{
			for (int j = i; j < labels.size(); j+=parallelNumber)
			{
				if(minusNumber.count(labels[j]))
				{
					labels[j]-=minusNumber[labels[j]];
				}
			}
		}
	}
	pair<vector<vector<double> >,vector<int> > result;
	
	result.first=trueCenters;
	result.second=labels;

	return result;

}