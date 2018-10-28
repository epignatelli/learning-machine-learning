#pragma once
#include <xtensor\xarray.hpp>

namespace Learning {

	class Losses
	{
		/**********************************
		/*** Public Methods
		/**********************************/
	public:
		static xt::xarray<double> NegLogLikelihood(xt::xarray<double> yhat, xt::xarray<double> y);


		/**********************************
		/*** Private Constructors
		/**********************************/
	private:
		// Constructor
		Losses();

		// Destructor
		~Losses();
	};
}
