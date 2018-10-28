#pragma once

//#include "Model.h"
#include "xtensor\xarray.hpp"

namespace Learning {

	class LogisticRegressionModel
	{
		/**********************************
		/*** Members
		/**********************************/
	public:
		xt::xarray<double> w;

		xt::xarray<double> b;

		/**********************************
		/*** Methods
		/**********************************/
	public:
		virtual void Initialise(std::vector<double> shape);

		virtual void Propagate();

		virtual void Backpropagate();

		virtual void Update();

		virtual void Train();

		virtual void Predict();


		/**********************************
		/*** Constructors
		/**********************************/
	public:
		LogisticRegressionModel();

		~LogisticRegressionModel();
	};
}
