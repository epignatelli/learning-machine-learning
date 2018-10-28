#pragma once
#include <xtensor\xarray.hpp>

namespace Learning {

	class Activations
	{
		/**********************************
		/*** Public Methods
		/**********************************/
	public:
		static xt::xarray<double> Sigmoid(xt::xarray<double> z);

		static xt::xarray<double> ReLU(xt::xarray<double> z);

		static xt::xarray<double> Tanh(xt::xarray<double> z);


		/**********************************
		/*** Private Constructors
		/**********************************/
	private:
		// Constructor
		Activations();

		// Destructor
		~Activations();
	};
}
