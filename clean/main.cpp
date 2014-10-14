#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "..\FileIO\FileInOut.h"

#include <vector>
#include <string>
#include <windows.h>
#include <direct.h>
#include <string>
#include <unordered_map>
#include <algorithm>
using namespace std;
using namespace cv;


#define _NUM_OF_SIMILAR_CODES 30

template<class T>
static void FromSmall(vector<T>& p,int n,vector<int>& index)
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

vector<vector<double> > imageToFeatures(string s)
{
	
	Mat iptimg=imread(s);

	SiftFeatureDetector detector;
	vector<KeyPoint> temkpts;
	detector.detect(iptimg,temkpts);
	SiftDescriptorExtractor extractor;
	Mat descriptor;
	extractor.compute(iptimg,temkpts,descriptor);
	
	
	vector<vector<double> > features;
	features.resize(descriptor.rows,vector<double>(128,0.0));
	for (int i=0;i<descriptor.rows;i++)
	{

		

		double temsum(0.000001);
		for (int j=0;j<128;j++)
		{
			temsum+=descriptor.at<float>(i,j);
				
		}
		for (int j=0;j<128;j++)
		{
			
			features[i][j]=descriptor.at<float>(i,j)/temsum;
		}


	}
	return features;
}

vector<vector<double> > imageToFeaturesQuick(string s)
{
	return fileIOclass::InVectorSDouble(s+".sift");
}


inline bool fileExists (const string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}


pair<vector<vector<double> >,vector<string> > train(string folder)
{
	pair<vector<vector<double> >,vector<string> > result;

	_chdir(folder.c_str());
	vector<string> subFolderList=fileIOclass::InVectorString("folder.lst");

	for(int i=0;i<subFolderList.size();++i)
	{
		string subfolder=folder+subFolderList[i];
		cout<<subfolder<<endl;
		_chdir(subfolder.c_str());
		vector<string> imageList=fileIOclass::InVectorString("allimg.lst");
		//#pragma omp parallel for
		for(int j=0;j<imageList.size();++j)
		{
			vector<vector<double> > features=imageToFeaturesQuick(imageList[j]);
			//cout<<imageList[j] <<" ";
			result.first.insert(result.first.end(),features.begin(),features.end());
			vector<string> labels(features.size(),subFolderList[i]);
			result.second.insert(result.second.end(),labels.begin(),labels.end());
		}

	}

	return result;
}

double dis(vector<double>& a,vector<double>& b)
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

vector< pair<vector<string>,vector<int> > > test(string folder,pair<vector<vector<double> >,vector<string> >& codebook)
{
	cout<<"codebook size"<<codebook.first.size()<<endl;
	assert(codebook.first.size()==codebook.second.size());
	assert(codebook.first.size()>_NUM_OF_SIMILAR_CODES);

	
	//for(int i=0;i<codebook.first.size();++i)
		



	_chdir(folder.c_str());

	vector<string> imageList=fileIOclass::InVectorString("allimg.lst");
	vector<pair<vector<string>,vector<int>> > result(imageList.size());
//	result.resize(imageList.size(),"");
	#pragma omp parallel for
	for(int i=0;i<imageList.size();++i)
	{
		string fileName=imageList[i]+".label";

		if(!fileExists(fileName))
		{
			vector<vector<double> > features=imageToFeaturesQuick(imageList[i]);
			//if (i%100==0)
			
			unordered_map<string,int> stastics;
			for(int j=0;j<features.size();++j)
			{
				vector<int> index(codebook.first.size());
				vector<double> distances(codebook.first.size(),0.0);
				for(int k=0;k<codebook.first.size();++k)
				{
					distances[k] = dis(features[j],codebook.first[k]);
					index[k]=k;
				}
				//vector<int> temindex=index;
				FromSmall(distances,index.size(),index);

				for(int k=0;k<_NUM_OF_SIMILAR_CODES;++k)
				{
					//cout<<index[k]<<endl;
					string label=codebook.second[index[k]];
					if(stastics.count(label))
						++stastics[label];
					else
						stastics[label]=1;
				}

			}

			pair<vector<string>,vector<int> > orders;
			orders.first.resize(stastics.size(),"");
			orders.second.resize(stastics.size(),0);

			vector<string> labelorder(stastics.size());
			vector<int> numberorder(stastics.size());
			vector<int> subindex(stastics.size());
			int j=0;
			for(unordered_map<string,int>::iterator it=stastics.begin();it!=stastics.end();++it)
			{
				labelorder[j]=it->first;
				numberorder[j]=it->second;
				subindex[j]=j;
				++j;
			}
			vector<int> temnumberorder=numberorder;
			FromSmall(temnumberorder,subindex.size(),subindex);
			for(int k=0;k<subindex.size();++k)
			{
				int _ind=subindex.size()-1-k;
				orders.first[k]=labelorder[subindex[_ind]];
				orders.second[k]=numberorder[subindex[_ind]];
			}
			result[i]=orders;

		
			FILE* fp=fopen(fileName.c_str(),"w");
			fprintf(fp,"%d\n",orders.first.size());
			for(int k=0;k<orders.first.size();++k)
			{
				fprintf(fp,"%s %d\n",orders.first[k].c_str(),orders.second[k]);
			}

			fclose(fp);
			cout<<"finished number "<<i<<endl;
		}
	}
	
	return result;
}


int mainsift(int argc,char* argv[])
{
	string s=argv[1];

	vector<vector<double> > features=imageToFeatures(s);

	fileIOclass::OutVectorSDouble(s+".sift",features);
	return 0;
}

int main_()
{
	string trainingFolder="D:\\ZPWang\\Ex01\\training\\";
	string testingFolder="D:\\ZPWang\\Ex01\\input\\";
	
	cout<<"training...."<<endl;
	pair<vector<vector<double>>,vector<string>> codebook=train(trainingFolder);
	_chdir(trainingFolder.c_str());

	fileIOclass::OutVectorSDouble("features.txt",codebook.first);
	fileIOclass::OutVectorString("labels.txt",codebook.second);
	cout<<"finished training...\ntesting...\n"<<endl;

	vector<pair<vector<string>,vector<int> > > labels=test(testingFolder,codebook);
	_chdir(testingFolder.c_str());
	//fileIOclass::OutVectorString("lables.txt",labels);


	
    return 0;

}

int main()
{
	string trainingFolder="D:\\ZPWang\\Ex01\\training\\";
	string testingFolder="D:\\ZPWang\\Ex01\\input\\";
	
	cout<<"training...."<<endl;
	pair<vector<vector<double>>,vector<string>> codebook;//=train(trainingFolder);
	_chdir(trainingFolder.c_str());

	//fileIOclass::OutVectorSDouble("features.txt",codebook.first);
	//fileIOclass::OutVectorString("labels.txt",codebook.second);

	codebook.first=fileIOclass::InVectorSDouble("features.txt");
	codebook.second=fileIOclass::InVectorString("labels.txt");

	cout<<"finished training...\ntesting...\n"<<endl;

	vector<pair<vector<string>,vector<int> > > labels=test(testingFolder,codebook);
	_chdir(testingFolder.c_str());
	//fileIOclass::OutVectorString("lables.txt",labels);


	
    return 0;

}


