#include "utinity.h"
#define _NUM_OF_SIMILAR_CODES 30

#include "settings.inl"
using namespace FujitsuTask;

#include <algorithm>

vector<vector<double> > imageToFeaturesQuick(const string& s)
{
	return fileIOclass::InVectorSDouble(s);
}

class trainTestTask
{
public:
	string taskName;
	string trainFolder;
	string testFolder;
	string trainImages;
	string trainLabels;
	string testImages;


	string feaFileName;//=taskName+"features.txt";
	string labelFieName;//=taskName+"classlabel.txt";
	string centerFileName;//=taskName+"featureCenters.txt";
	string clusterLabelFileName;//=taskName+"clusterlabels";

	vector<vector<double> > features;
	vector<string> labels;

	bool eachCategoryMode;

	trainTestTask(const string& _taskName,const string& _trainFolder,const string& _testFolder,const string& _trainImages,const string& _trainLabels,const string& _testImages,bool _eachCategoryMode=true)
		:taskName(_taskName),trainFolder(_trainFolder),testFolder(_testFolder),trainImages(_trainImages),trainLabels(_trainLabels),testImages(_testImages),eachCategoryMode(_eachCategoryMode)
	{
		feaFileName=taskName+"features.txt";
		labelFieName=taskName+"classlabel.txt";
		centerFileName=taskName+"featureCenters.txt";
		clusterLabelFileName=taskName+"clusterlabels";
	}

	void generateGroundTrueth(const string& totalList,const string& totalLabel,string classlist="",string categoryList="")
	{
		_chdir(trainFolder.c_str());
		auto tlist=fileIOclass::InVectorString(totalList);
		auto tlables=fileIOclass::InVectorString(totalLabel);
		
		assert(tlist.size()==tlables.size());

		auto imgNames=fileIOclass::InVectorString(trainImages);
		auto labelNames=fileIOclass::InVectorString(trainLabels);

		unordered_set<string> _trainImgs;
		unordered_set<string> _trainLables;

		vector<string> groundThruthNames;
		vector<string> groundThrthLabels;

		for(auto s:imgNames)
			_trainImgs.insert(s);


		if(eachCategoryMode)
		{
			auto classNames=fileIOclass::InVectorString(classlist);
			auto categoryNames=fileIOclass::InVectorString(categoryList);
			assert(classNames.size()==categoryNames.size());
			unordered_map<string,string> classCategoryCorrespondences;
			for (int i = 0; i < classNames.size(); i++)
			{
				classCategoryCorrespondences[classNames[i]]=categoryNames[i];
			}
			for(auto s:labelNames)
				_trainLables.insert(classCategoryCorrespondences[s]);

			for(int i=0;i<tlist.size();++i)
			{
				if( (!_trainImgs.count(tlist[i]) )&& _trainLables.count(classCategoryCorrespondences[tlables[i]]))
				{
					groundThruthNames.push_back(tlist[i]);
					groundThrthLabels.push_back(tlables[i]);
				}
			}

		}
		else
		{
			for(auto s:labelNames)
				_trainLables.insert(s);

			for(int i=0;i<tlist.size();++i)
			{
				if( (!_trainImgs.count(tlist[i]) )&& _trainLables.count(tlables[i]))
				{
					groundThruthNames.push_back(tlist[i]);
					groundThrthLabels.push_back(tlables[i]);
				}
			}
		}

		fileIOclass::OutVectorString(taskName+"groundTruth"+"_names.txt",groundThruthNames);
		fileIOclass::OutVectorString(taskName+"groundTruth"+"_labels.txt",groundThrthLabels);


	}



	void train()
	{
		_chdir(trainFolder.c_str());
		if(!( fileExists(feaFileName)&& fileExists(labelFieName)) )
		{
			features.clear();
			labels.clear();
			
			auto imgNames=fileIOclass::InVectorString(trainImages);
			auto labelNames=fileIOclass::InVectorString(trainLabels);

			for(int i=0;i<imgNames.size();i++)
			{
				if(i%3==0) cout<<i<<endl;

				auto tfeatures=imageToFeaturesQuick(imgNames[i]+".jpg.sift");
				vector<string> tlabels=vector<string>(tfeatures.size(),labelNames[i]);
				features.insert(features.end(),tfeatures.begin(),tfeatures.end());
				labels.insert(labels.end(),tlabels.begin(),tlabels.end());
			}
			fileIOclass::OutVectorSDouble(feaFileName,features);
			fileIOclass::OutVectorString(labelFieName,labels);
			cout<<"codebook generated<<"<<endl;
		}
	}

	
	
	void doKmeans()
	{
		_chdir(trainFolder.c_str());
		if(!fileExists(centerFileName))
		{
			if(features.size()==0)
			{
				features=fileIOclass::InVectorSDouble(feaFileName);
				labels=fileIOclass::InVectorString(labelFieName);
			}
			auto clusters=parallelKMeans2(features);
			fileIOclass::OutVectorSDouble(centerFileName,clusters.first);
			fileIOclass::OutVectorInt(clusterLabelFileName,clusters.second);
		}
	}

	void testForList()
	{
		_chdir(trainFolder.c_str());
		if (features.size()==0)
		{
			if( !fileExists(feaFileName)||
				!fileExists(labelFieName)||
				!fileExists(centerFileName)||
				!fileExists(clusterLabelFileName)
				)
			{
				cout<<"training not finished for "<<taskName<<endl;
			}
			else
			{
				_chdir(trainFolder.c_str());
				features=fileIOclass::InVectorSDouble(feaFileName);
				labels=fileIOclass::InVectorString(labelFieName);
				pair<vector<vector<double> >,vector<int> > clusters;
				clusters.first=fileIOclass::InVectorSDouble(centerFileName);
				clusters.second=fileIOclass::InVectorInt(clusterLabelFileName);

				_chdir(testFolder.c_str());
				auto testList=fileIOclass::InVectorString(testImages);

				vector<string> testlabel(testList.size());
				for (int i = 0; i < testList.size(); i++)
				{
					testlabel[i]=testForOne(testList[i],clusters);
				}

				fileIOclass::OutVectorString(testImages+".rslt",testlabel);				
			}
		}
	}
	void evaluateResultByGroundtruth(const string& result,const string& groudtruth,const string& classname="",const string& categoryname="")
	{
		_chdir(testFolder.c_str());
		auto lisResult=fileIOclass::InVectorString(result);
		auto listGtruth=fileIOclass::InVectorString(groudtruth);
		evaluateResultByGroudtruthCore(lisResult,listGtruth,result,groudtruth);

		if(classname!="")
		{
			auto lisclass=fileIOclass::InVectorString(classname);
			auto liscategory=fileIOclass::InVectorString(categoryname);
			assert(classname.size()==categoryname.size());

			unordered_map<string,string> dict;
			for (int i = 0; i < classname.size(); i++)
			{
				dict[lisclass[i]]=liscategory[i];
			}

			vector<string> dlisresult(lisResult.size());
			vector<string> dlisgtruth(listGtruth.size());
			for (int i = 0; i < lisResult.size(); i++)
			{
				dlisresult[i]=dict[lisResult[i]];
				dlisgtruth[i]=dict[listGtruth[i]];
			}
			evaluateResultByGroudtruthCore(dlisresult,dlisgtruth,result+"category",groudtruth+"category");
		}
		
	}
	void evaluateResultByGroudtruthCore(const vector<string>& lisResult,const vector<string>& listGtruth,const string& result,const string& groudtruth)
	{
		assert(lisResult.size()==listGtruth.size());
		unordered_set<string> resultD;
		unordered_set<string> gtruthD;
		for (int i = 0; i < lisResult.size(); i++)
		{
			resultD.insert(lisResult[i]);
			gtruthD.insert(listGtruth[i]);
		}

		vector<string> resultV;
		vector<string> gtruthV;
		for (auto it=resultD.begin();it!=resultD.end();++it )
		{
			resultV.push_back(*it);
		}
		sort(resultV.begin(),resultV.end());

		for(auto it=gtruthD.begin();it!=gtruthD.end();++it)
		{
			gtruthV.push_back(*it);
		}
		sort(gtruthV.begin(),gtruthV.end());

		unordered_map<string,int> resultIndex;
		for (int i = 0; i < resultV.size(); i++)
		{
			resultIndex[resultV[i]]=i;
		}

		unordered_map<string,int> gtruthIndex;
		for (int i = 0; i < gtruthV.size(); i++)
		{
			gtruthIndex[gtruthV[i]]=i;
		}


		vector<vector<int> > matrix( resultV.size(),vector<int> (gtruthV.size(),0));

		for (int i = 0; i < lisResult.size(); i++)
		{
			++matrix[resultIndex[lisResult[i]]] [gtruthIndex[listGtruth[i]]];
		}
		fileIOclass::OutVectorString(result+groudtruth+"result",resultV);
		fileIOclass::OutVectorString(result+groudtruth+"gtruth",gtruthV);
		fileIOclass::OutVectorSInt(result+groudtruth+"_matrix",matrix);
	}

	private:
		string testForOne(string _testImage,const pair<vector<vector<double> >,vector<int> >& kclusters)
		{
			cout<<_testImage<<endl;
			auto tfeatures=imageToFeaturesQuick(_testImage+".jpg.sift");
			
			assert(features.size()>0);
			assert(labels.size()>0);
			assert(kclusters.first.size()>0);
			assert(kclusters.second.size()>0);

			vector<vector<string> > collectVotes(tfeatures.size(),vector<string>(_NUM_OF_SIMILAR_CODES,""));

//			vector<vector<double> > rankBuffer(tfeatures.size(), vector<double>(kclusters.first.size(),0.0));
			
			vector<double> disK(tfeatures.size(),0.0);
			vector<double> tdisK(tfeatures.size(),0.0);
			vector<int> bestK(tfeatures.size(),-1);

			#pragma omp parallel for
			for (int i = 0; i <parallelNumber; i++)
			{
				for (int j = i; j < tfeatures.size(); j+=parallelNumber)
				{
					disK[j]=dis(tfeatures[j],kclusters.first[0]);
					bestK[j]=0;
					for (int k = 1; k < kclusters.first.size(); k++)
					{
						tdisK[j]=dis(tfeatures[j],kclusters.first[k]);
						if(tdisK[j]<disK[j])
						{
							bestK[j]=k;
							disK[j]=tdisK[j];
						}

					}
				}
			}

			unordered_map<int,vector<int>> clusterContains;

			for (int i = 0; i < kclusters.second.size(); i++)
			{
				clusterContains[kclusters.second[i]].push_back(i);
			}

			vector<vector<double> > rankBuffer(tfeatures.size());
			vector<vector<int> > rankIndexBuffer(tfeatures.size());
			//#pragma omp parallel for
			for (int i = 0; i < rankBuffer.size(); i++)
			{
				rankBuffer[i].resize( clusterContains[bestK[i]].size(),0.0);
				rankIndexBuffer[i].resize( clusterContains[bestK[i]].size(),0);
				for (int j = 0; j < rankIndexBuffer[i].size(); j++)
				{
					rankIndexBuffer[i][j]=clusterContains[bestK[i]][j];
				}

			}

			

			#pragma omp parallel for
			for (int i = 0; i < parallelNumber; i++)
			{
				for (int j = i; j < tfeatures.size(); j+=parallelNumber)
				{
					for (int k = 0; k < clusterContains[ bestK[j]].size(); k++)
					{
						rankBuffer[j][k]=dis(tfeatures[j],features[clusterContains[bestK[j]][k]]);
					}
					FromSmall(rankBuffer[j],rankBuffer[j].size(),rankIndexBuffer[j]);
					for (int k = 0; k < rankBuffer[j].size() && k<_NUM_OF_SIMILAR_CODES; k++)
					{
						collectVotes[j][k]=labels[rankIndexBuffer[j][k]];
					}
				}
			}

			unordered_map<string,int> allvotes;
			for (int i = 0; i < collectVotes.size(); i++)
			{
				for (int j = 0; j < collectVotes[i].size(); j++)
				{
					if(collectVotes[i][j]!="")
					{
						if(!allvotes.count(collectVotes[i][j]))
						{
							allvotes[collectVotes[i][j]]=1;
						}
						else
							++allvotes[collectVotes[i][j]];
					}
				}
			}

			vector<string> labelorder(allvotes.size());
			vector<int> numberorder(allvotes.size());
			vector<int> subindex(allvotes.size());
			int j=0;
			for(unordered_map<string,int>::iterator it=allvotes.begin();it!=allvotes.end();++it)
			{
				labelorder[j]=it->first;
				numberorder[j]=it->second;
				subindex[j]=j;
				++j;
			}
			vector<int> temnumberorder=numberorder;

			FILE* fp=fopen((taskName+_testImage+".rslt").c_str(),"w");

			FromSmall(temnumberorder,subindex.size(),subindex);
			fprintf(fp,"%d\n",subindex.size());
			for(int k=0;k<subindex.size();++k)
			{
				int _ind=subindex.size()-1-k;
				fprintf(fp,"%s %d\n",labelorder[subindex[_ind]].c_str(),numberorder[subindex[_ind]]);
				//orders.first[k]=labelorder[subindex[_ind]];
				//orders.second[k]=numberorder[subindex[_ind]];
			}
			fclose(fp);
			return labelorder[subindex[subindex.size()-1]];
		}
};

