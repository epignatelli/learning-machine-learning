/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// This file is generated from test/files/cppy_source/test_extended_xmath_interp.cppy by preprocess.py!

#include <algorithm>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    using namespace xt::placeholders;

    /*py
    xp = np.sort(np.random.random(20) - 0.5)
    fp = np.random.random(20) - 0.5
    x  = np.linspace(-1,1,50)
    f  = np.interp(x, xp, fp)
    */
    TEST(xtest_extended_xmath, interp)
    {
        // py_xp
        xarray<double> py_xp = {-0.4392965543332054,-0.275018267626908 ,-0.2728014243888024,
                                -0.2696473197142562,-0.0928505276610206,-0.0204745699840858,
                                 0.080583493569036 , 0.0813849275417244, 0.1055683399973824,
                                 0.1438052604943405, 0.1948168233505687, 0.2095481035362423,
                                 0.2159819154499707, 0.2318449215017366, 0.2424323580651068,
                                 0.2933571131116369, 0.2935623420286422, 0.3129653486721635,
                                 0.3895138903219566, 0.4478872635644447};
        // py_fp
        xarray<double> py_fp = {-0.4208180015020071,-0.3857126093818573, 0.1880201888468721,
                                -0.0055635997524461,-0.103541672732925 , 0.302986148541293 ,
                                 0.0739468861216267, 0.1480935624943865, 0.1058657498049309,
                                 0.4258776883799703,-0.3429085458230178, 0.3191822568916822,
                                -0.1933872058961967, 0.0288436789118078,-0.2817737863989301,
                                 0.2802327222845201,-0.1471681181599579,-0.3096276961693655,
                                 0.0801296586980623,-0.4830695323894567};
        // py_x
        xarray<double> py_x = {-1.                ,-0.9591836734693877,-0.9183673469387755,
                               -0.8775510204081632,-0.8367346938775511,-0.7959183673469388,
                               -0.7551020408163265,-0.7142857142857143,-0.6734693877551021,
                               -0.6326530612244898,-0.5918367346938775,-0.5510204081632654,
                               -0.5102040816326531,-0.4693877551020409,-0.4285714285714286,
                               -0.3877551020408164,-0.3469387755102041,-0.3061224489795918,
                               -0.2653061224489797,-0.2244897959183674,-0.1836734693877552,
                               -0.1428571428571429,-0.1020408163265307,-0.0612244897959184,
                               -0.0204081632653061, 0.0204081632653061, 0.0612244897959182,
                                0.1020408163265305, 0.1428571428571428, 0.1836734693877551,
                                0.2244897959183672, 0.2653061224489794, 0.3061224489795917,
                                0.346938775510204 , 0.3877551020408163, 0.4285714285714284,
                                0.4693877551020407, 0.510204081632653 , 0.5510204081632653,
                                0.5918367346938773, 0.6326530612244896, 0.6734693877551019,
                                0.7142857142857142, 0.7551020408163265, 0.7959183673469385,
                                0.8367346938775508, 0.8775510204081631, 0.9183673469387754,
                                0.9591836734693877, 1.                };
        // py_f
        xarray<double> py_f = {-0.4208180015020071,-0.4208180015020071,-0.4208180015020071,
                               -0.4208180015020071,-0.4208180015020071,-0.4208180015020071,
                               -0.4208180015020071,-0.4208180015020071,-0.4208180015020071,
                               -0.4208180015020071,-0.4208180015020071,-0.4208180015020071,
                               -0.4208180015020071,-0.4208180015020071,-0.4185260994317218,
                               -0.409803868536272 ,-0.4010816376408223,-0.3923594067453725,
                               -0.0079694247537948,-0.0305892074612051,-0.0532089901686153,
                               -0.0758287728760256,-0.0984485555834358, 0.0740983154828471,
                                0.3028356435211665, 0.2103290092698822, 0.1178223750185983,
                                0.1120253275362987, 0.4179427145605969,-0.1749690307446933,
                               -0.0741970726917779,-0.0293385130219197,-0.2523327306601354,
                               -0.136647353307242 , 0.0711745479480076,-0.2967060923659397,
                               -0.4830695323894567,-0.4830695323894567,-0.4830695323894567,
                               -0.4830695323894567,-0.4830695323894567,-0.4830695323894567,
                               -0.4830695323894567,-0.4830695323894567,-0.4830695323894567,
                               -0.4830695323894567,-0.4830695323894567,-0.4830695323894567,
                               -0.4830695323894567,-0.4830695323894567};

        auto f = xt::interp(py_x, py_xp, py_fp);

        EXPECT_TRUE(xt::allclose(f, py_f));
    }
}

