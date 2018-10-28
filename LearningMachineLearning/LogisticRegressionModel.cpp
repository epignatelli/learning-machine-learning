#include "stdafx.h"
#include "LogisticRegressionModel.h"
#include <xtensor\xrandom.hpp>
#include <xtensor-blas\xlinalg.hpp>
#include "Activations.h"
#include "Losses.h"


namespace Learning {

	/**********************************
	/*** Constructors
	/**********************************/
	LogisticRegressionModel::LogisticRegressionModel(std::vector<double> shape) {
		(*this).w = xt::random::randn<double>(shape);
		(*this).b = xt::random::randn<double>({ shape[0], 1.0 });
		return;
	}

	/**********************************/

	LogisticRegressionModel::LogisticRegressionModel(xt::xarray<double> w, xt::xarray<double> b) {
		(*this).w = w;
		(*this).b = b;
		return;
	}

	/**********************************/

	LogisticRegressionModel::~LogisticRegressionModel()
	{
		//delete (*this).w;
		//delete (*this).b;
	}


	/**********************************
	/*** Methods
	/**********************************/

	xt::xarray<double> LogisticRegressionModel::Propagate(xt::xarray<double> X) {
		xt::xarray<double> z = xt::linalg::dot(xt::transpose(w), X) + b;
		return Activations::Sigmoid(z);
	}

	/**********************************/

	std::tuple<xt::xarray<double>, xt::xarray<double>> LogisticRegressionModel::Backpropagate(xt::xarray<double> X, xt::xarray<double> yhat, xt::xarray<double> y) {
		int m = X.shape[1];

		xt::xarray<double> dw = xt::linalg::dot(X, xt::transpose(yhat - y)) / m;
		xt::xarray<double> db = xt::sum(yhat - y) / m;
		return std::tuple<xt::xarray<double>, xt::xarray<double>>(dw, db);
	}

	/**********************************/

	void LogisticRegressionModel::Update(xt::xarray<double> dw, xt::xarray<double> db, double alpha=0.01) {
		(*this).w -= alpha * dw;
		(*this).b -= alpha * db;
		return;
	}
	
	/**********************************/

	void LogisticRegressionModel::Train(xt::xarray<double> X, xt::xarray<double> y, double alpha=0.01, int iterations=100) {
		int m = X.shape[1];
		double* costs = new double[iterations];

		for (int i = 0; i < iterations; i++) {
			xt::xarray<double> a = LogisticRegressionModel::Propagate(X);

			xt::xarray<double> losses = Losses::NegLogLikelihood(a, y);
			costs[i] = xt::sum(losses, { 1 })() / m;
			std::cout << costs[i] << std::endl;

			std::tuple<xt::xarray<double>, xt::xarray<double>> grads = LogisticRegressionModel::Backpropagate(X, a, y);
			xt::xarray<double> dw = std::get<0>(grads);
			xt::xarray<double> db = std::get<1>(grads);

			LogisticRegressionModel::Update(dw, db);
		}
		return;
	}
	
	/**********************************/
}

