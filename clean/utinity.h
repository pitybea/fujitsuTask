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
using namespace std;

#pragma once 


pair<vector<vector<double> >,vector<int> > parallelKMeans(const vector<vector<double> >& dataset,int kCenter=-1);
