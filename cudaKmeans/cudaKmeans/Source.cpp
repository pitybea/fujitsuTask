#include  "main.h"
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "..\FileIO\FileInOut.h"

#include <direct.h>
#include <Windows.h>

int main()
{
	
	int    *membership;    /* [numObjs] */
	char   *filename;
	float **objects;       /* [numObjs][numCoords] data objects */
	float **clusters;      /* [numClusters][numCoords] cluster center */
	float   threshold;
	int     loop_iterations;
	int     numClusters, numCoords, numObjs;

	threshold        = 0.00001;
   
   
	_chdir("D:\\DATA\\Fujitsu\\images\\training");
	auto data=fileIOclass::InVectorSDouble("task08_features.txt");

	 numClusters      = data.size()/1000;
	numObjs=data.size();
	numCoords=data[0].size();

	objects=new float*[numObjs];
	for(int i=0;i<numObjs;++i)
	{
		objects[i]=new float[numCoords];
		for (int j = 0; j < numCoords; j++)
		{
			objects[i][j]=data[i][j];
		}
	}


	membership =new int[numObjs];//= (int*) malloc(numObjs * sizeof(int));
	clusters = cuda_kmeans(objects, numCoords, numObjs, numClusters, threshold,
                          membership, &loop_iterations);

	for(int i=0;i<numObjs;++i)
		delete[] objects[i];

    delete[] objects;

	vector<int> lbs(numObjs,0);
	for(int i=0;i<numObjs;++i)
		lbs[i]=membership[i];

	fileIOclass::OutVectorInt("task08_clusterlabels",lbs);

	delete[] membership;

	vector<vector<double> > centers(numClusters,vector<double>(numCoords,0.0));

	for(int i=0;i<numClusters;++i)
	{
		for (int j = 0; j < numCoords; j++)
		{
			centers[i][j]=clusters[i][j];
		}
	   
	}

	free(clusters[0]);

    free( clusters);
	fileIOclass::OutVectorSDouble("task08_featureCenters.txt",centers);

	return 0;
}