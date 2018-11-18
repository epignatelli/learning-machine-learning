#pragma once

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
		xt::xarray<double> Propagate(xt::xarray<double> X);

		std::tuple<xt::xarray<double>, xt::xarray<double>> Backpropagate(xt::xarray<double> X, xt::xarray<double> yhat, xt::xarray<double> y);

		void Update(xt::xarray<double> dw, xt::xarray<double> db, double alpha);

		void Train(xt::xarray<double> X, xt::xarray<double> Y, double alpha, int iterations);

		//virtual void Predict();


		/**********************************
		/*** Constructors
		/**********************************/
	public:
		LogisticRegressionModel(std::vector<double> shape);

		LogisticRegressionModel(xt::xarray<double> w, xt::xarray<double> b);

		~LogisticRegressionModel();
	};
}
