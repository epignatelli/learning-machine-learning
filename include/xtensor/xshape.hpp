/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

#include "xexception.hpp"
#include "xstorage.hpp"

namespace xt
{
    template <class T>
    using dynamic_shape = svector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    using xindex = dynamic_shape<std::size_t>;
}

namespace xtl
{
    namespace detail
    {
        template <class S>
        struct sequence_builder;

        template <std::size_t... I>
        struct sequence_builder<xt::fixed_shape<I...>>
        {
            using sequence_type = xt::fixed_shape<I...>;
            using value_type = typename sequence_type::value_type;

            inline static sequence_type make(std::size_t /*size*/)
            {
                return sequence_type{};
            }

            inline static sequence_type make(std::size_t /*size*/, value_type /*v*/)
            {
                return sequence_type{};
            }
        };
    }
}

namespace xt
{

    /***********************************
     * static_dimension implementation *
     ***********************************/

    namespace detail
    {
        template <class T, class E = void>
        struct static_dimension_impl
        {
            static constexpr std::ptrdiff_t value = -1;
        };

        template <class T>
        struct static_dimension_impl<T, void_t<decltype(std::tuple_size<T>::value)>>
        {
            static constexpr std::ptrdiff_t value = static_cast<std::ptrdiff_t>(std::tuple_size<T>::value);
        };
    }

    template <class S>
    struct static_dimension
    {
        static constexpr std::ptrdiff_t value = detail::static_dimension_impl<S>::value;
    };

    /*************************************
     * promote_shape and promote_strides *
     *************************************/

    namespace detail
    {
        template <class T1, class T2>
        constexpr std::common_type_t<T1, T2> imax(const T1& a, const T2& b)
        {
            return a > b ? a : b;
        }

        // Variadic meta-function returning the maximal size of std::arrays.
        template <class... T>
        struct max_array_size;

        template <>
        struct max_array_size<>
        {
            static constexpr std::size_t value = 0;
        };

        template <class T, class... Ts>
        struct max_array_size<T, Ts...> : std::integral_constant<std::size_t, imax(std::tuple_size<T>::value, max_array_size<Ts...>::value)>
        {
        };

        // Broadcasting for fixed shapes
        template <std::size_t IDX, std::size_t... X>
        struct at
        {
            constexpr static std::size_t arr[sizeof...(X)] = {X...};
            constexpr static std::size_t value = (IDX < sizeof...(X)) ? arr[IDX] : 0;
        };

        template <class S1, class S2>
        struct broadcast_fixed_shape;

        template <class IX, class A, class B>
        struct broadcast_fixed_shape_impl;

        template <std::size_t IX, class A, class B>
        struct broadcast_fixed_shape_cmp_impl;

        template <std::size_t JX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>
        {
            //We line the shapes up from the last index
            //IX may underflow, thus being a very large number
            static constexpr std::size_t IX = JX - (sizeof...(J) - sizeof...(I));

            //Out of bounds access gives value 0
            static constexpr std::size_t I_v = at<IX, I...>::value;
            static constexpr std::size_t J_v = at<JX, J...>::value;

            // we're statically checking if the broadcast shapes are either one on either of them or equal
            static_assert(!I_v ||  I_v == 1 || J_v == 1 || J_v == I_v, "broadcast shapes do not match.");

            static constexpr std::size_t ordinate = (I_v > J_v) ? I_v : J_v;
            static constexpr bool value = (I_v == J_v);
        };

        template <std::size_t... JX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_impl<std::index_sequence<JX...>, fixed_shape<I...>, fixed_shape<J...>>
        {
            static_assert(sizeof... (J) >= sizeof... (I), "broadcast shapes do not match.");

            using type = xt::fixed_shape<broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>::ordinate...>;
            static constexpr bool value = xtl::conjunction<broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>...>::value;
        };

        /* broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>
         * Just like a call to broadcast_shape(cont S1& input, S2& output),
         * except that the result shape is alised as type, and the returned
         * bool is the member value. Asserts on an illegal broadcast, including
         * the case where pack I is strictly longer than pack J. */

        template <std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>
            : broadcast_fixed_shape_impl<std::make_index_sequence<sizeof...(J)>, fixed_shape<I...>, fixed_shape<J...>> {};

        // Simple is_array and only_array meta-functions
        template <class S>
        struct is_array
        {
            static constexpr bool value = false;
        };

        template <class T, std::size_t N>
        struct is_array<std::array<T, N>>
        {
            static constexpr bool value = true;
        };

        template <class S>
        struct is_fixed
        {
            static constexpr bool value = false;
        };

        template <std::size_t... N>
        struct is_fixed<fixed_shape<N...>>
        {
            static constexpr bool value = true;
        };

        template <class S>
        struct is_scalar_shape
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct is_scalar_shape<std::array<T, 0>>
        {
            static constexpr bool value = true;
        };

        template <class... S>
        using only_array = xtl::conjunction<xtl::disjunction<is_array<S>, is_fixed<S>>...>;

        // test that at least one argument is a fixed shape. If yes, then either argument has to be fixed or scalar
        template <class... S>
        using only_fixed = std::integral_constant<bool, xtl::disjunction<is_fixed<S>...>::value &&
                                                        xtl::conjunction<xtl::disjunction<is_fixed<S>, is_scalar_shape<S>>...>::value>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array

        template <class... S>
        struct promote_array
        {
            using type = std::array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };

        template <>
        struct promote_array<>
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class S>
        struct filter_scalar
        {
            using type = S;
        };

        template <class T>
        struct filter_scalar<std::array<T, 0>>
        {
            using type = fixed_shape<1>;
        };

        template <class S>
        using filter_scalar_t = typename filter_scalar<S>::type;

        template <class... S>
        struct promote_fixed : promote_fixed<filter_scalar_t<S>...> {};

        template <std::size_t... I>
        struct promote_fixed<fixed_shape<I...>>
        {
            using type = fixed_shape<I...>;
            static constexpr bool value = true;
        };

        template <std::size_t... I, std::size_t... J, class... S>
        struct promote_fixed<fixed_shape<I...>, fixed_shape<J...>, S...>
        {
        private:

            using intermediate = std::conditional_t< (sizeof... (I) > sizeof... (J)),
                broadcast_fixed_shape<fixed_shape<J...>, fixed_shape<I...>>,
                broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>>;
            using result = promote_fixed<typename intermediate::type, S...>;

        public:

            using type = typename result::type;
            static constexpr bool value = xtl::conjunction<intermediate, result>::value;
        };

        template <bool all_index, bool all_array, class... S>
        struct select_promote_index;

        template <class... S>
        struct select_promote_index<true, true, S...> : promote_fixed<S...> {};

        template <>
        struct select_promote_index<true, true>
        {
            // todo correct? used in xvectorize
            using type = dynamic_shape<std::size_t>;
        };

        template <class... S>
        struct select_promote_index<false, true, S...> : promote_array<S...> {};

        template <class... S>
        struct select_promote_index<false, false, S...>
        {
            using type = dynamic_shape<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index : select_promote_index<only_fixed<S...>::value, only_array<S...>::value, S...> {};

        template <class T>
        struct index_from_shape_impl
        {
            using type = T;
        };

        template <std::size_t... N>
        struct index_from_shape_impl<fixed_shape<N...>>
        {
            using type = std::array<std::size_t, sizeof...(N)>;
        };
    }

    template <class... S>
    using promote_shape_t = typename detail::promote_index<S...>::type;

    template <class... S>
    using promote_strides_t = typename detail::promote_index<S...>::type;

    template <class S>
    using index_from_shape_t = typename detail::index_from_shape_impl<S>::type;
}

#endif