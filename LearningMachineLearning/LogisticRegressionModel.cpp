#include "stdafx.h"
#include "LogisticRegressionModel.h"
#include "xtensor\xrandom.hpp"

namespace Learning {

	/**********************************
	/*** Constructors
	/**********************************/
	LogisticRegressionModel::LogisticRegressionModel()
	{
	}

	/**********************************/

	LogisticRegressionModel::~LogisticRegressionModel()
	{
	}


	/**********************************
	/*** Methods
	/**********************************/

	void LogisticRegressionModel::Initialise(std::vector<double> shape)
	{
		LogisticRegressionModel::w = xt::random::randn<double>(shape);
		LogisticRegressionModel::b = xt::random::randn<double>({shape[0], 1.0});

	}

	/**********************************/
}

