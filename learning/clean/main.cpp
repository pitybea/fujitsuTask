


#include "..\FileIO\FileInOut.h"

#include "trainTest.h"
using namespace std;



#define _NUM_OF_SIMILAR_CODES 30










pair<vector<vector<double> >,vector<string> > train(const string& folder)
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





int main2()
{
	_chdir("D:\\DATA\\Fujitsu\\images\\");
	string trainFolder="D:\\DATA\\Fujitsu\\images\\training\\";
	
	string testFolder="D:\\DATA\\Fujitsu\\images\\test\\";
	auto tasks=fileIOclass::InVectorString("task.lst");

	for(int i=0;i<tasks.size();++i)
	{
		cout<<"Task number " <<i<<endl;
		string trainimages=tasks[i]+"name.txt";
		string trainlabels=tasks[i]+"label.txt";
		string testimages=tasks[i]+"test.txt";
		trainTestTask oneTask(tasks[i],trainFolder,testFolder,trainimages,trainlabels,testimages);
		oneTask.train();
		//oneTask.doKmeans();
	}


}


int kmain(int argc,char* argv[])
{

	assert(argc>1);
	string task=argv[1];
	_chdir("D:\\DATA\\Fujitsu\\images\\");
	string trainFolder="D:\\DATA\\Fujitsu\\images\\training\\";
	
	string testFolder="D:\\DATA\\Fujitsu\\images\\test\\";
	auto tasks=fileIOclass::InVectorString("task.lst");

	//for(int i=0;i<tasks.size();++i)
	//{
		cout<<task<<endl;
		string trainimages=task+"name.txt";
		string trainlabels=task+"label.txt";
		string testimages=task+"test.txt";
		trainTestTask oneTask(task,trainFolder,testFolder,trainimages,trainlabels,testimages);
		oneTask.train();
		//oneTask.doKmeans();
	//}

	return 0;
}

int main1_()
{
	string trainingFolder="D:\\ZPWang\\Ex01\\training\\";
	string testingFolder="D:\\ZPWang\\Ex01\\input\\";
	
	cout<<"training...."<<endl;
	pair<vector<vector<double>>,vector<string>> codebook;//=train(trainingFolder);
	_chdir(trainingFolder.c_str());

	//fileIOclass::OutVectorSDouble("features.txt",codebook.first);
	//fileIOclass::OutVectorString("labels.txt",codebook.second);

	codebook.first=fileIOclass::InVectorSDouble("features.txt");
	//codebook.second=fileIOclass::InVectorString("labels.txt");

	cout<<"finished training...\ntesting...\n"<<endl;
	//cout<<"finished loading...\ntesting...\n"<<endl;


	auto cluster=parallelKMeans(codebook.first);
	fileIOclass::OutVectorSDouble("centers.txt",cluster.first);
	fileIOclass::OutVectorInt("clusterlables.txt",cluster.second);


	//vector<pair<vector<string>,vector<int> > > labels=test(testingFolder,codebook);
	//_chdir(testingFolder.c_str());
	//fileIOclass::OutVectorString("lables.txt",labels);


	
    return 0;

}
int main()
{
	_chdir("F:\\fujitsu");
	auto data=fileIOclass::InVectorSDouble("test.f");
	auto kmresult1=parallelKMeans(data,20);
	auto kmresult2=parallelKMeans2(data,20);

	fileIOclass::OutVectorSDouble("rc1.txt",kmresult1.first);
	fileIOclass::OutVectorSDouble("rc2.txt",kmresult2.first);

	fileIOclass::OutVectorInt("r1.txt",kmresult1.second);
	fileIOclass::OutVectorInt("r2.txt",kmresult2.second);

	return 0;
}
