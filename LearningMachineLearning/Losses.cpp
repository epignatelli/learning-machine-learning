#include "stdafx.h"
#include "Losses.h"

namespace Learning {

	/**********************************
	/*** Methods
	/**********************************/

	xt::xarray<double> Losses::NegLogLikelihood(xt::xarray<double> yhat, xt::xarray<double> y) {
		return y * xt::log(yhat) + (1 - y) * xt::log(1 - yhat);
	}


	/**********************************
	/*** Constructors
	/**********************************/
	Losses::Losses()
	{
	}

	/**********************************/

	Losses::~Losses()
	{
	}
	
	/**********************************/
}
