#ifndef TYPE_TRAITS_H_
#define TYPE_TRAITS_H_

template<class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

template<class T, class U>
struct is_same : false_type {};
 
template<class T>
struct is_same<T, T> : true_type {};

template< class T, class U >
constexpr bool is_same_v = is_same<T, U>::value;

#endif // TYPE_TRAITS_H_