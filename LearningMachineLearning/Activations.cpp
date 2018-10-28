#include "stdafx.h"
#include "Activations.h"


namespace Learning {

	/**********************************
	/*** Methods
	/**********************************/

	xt::xarray<double> Activations::Sigmoid(xt::xarray<double> z) {
		return 1 / xt::exp(-z);
	}

	/**********************************/

	xt::xarray<double> Activations::ReLU(xt::xarray<double> z) {
		return z * (z < 0);
	}

	/**********************************/

	xt::xarray<double> Activations::Tanh(xt::xarray<double> z) {
		xt::xarray<double> expZpos = xt::xarray<double>(z);
		xt::xarray<double> expZneg = xt::xarray<double>(-z);
		return (expZpos - expZneg) / (expZpos + expZneg);
	}


	/**********************************
	/*** Constructors
	/**********************************/
	Activations::Activations()
	{
	}

	/**********************************/

	Activations::~Activations()
	{
	}

	/**********************************/
}
