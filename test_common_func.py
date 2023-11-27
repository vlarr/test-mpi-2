from unittest import TestCase

import numpy as np
import numpy.testing

import common_func as cf


class Test(TestCase):
    def test_pow2_dist(self):
        np.testing.assert_almost_equal(cf.pow2_dist([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), 1.0)
        np.testing.assert_almost_equal(cf.pow2_dist([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 3.0)
        np.testing.assert_almost_equal(cf.pow2_dist([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), 14.0)
        np.testing.assert_almost_equal(cf.pow2_dist([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]), 1.0)
        np.testing.assert_almost_equal(cf.pow2_dist([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), 3.0)
        np.testing.assert_almost_equal(cf.pow2_dist([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 14.0)

    def test_calc_rv(self):
        def test(v_i, v_j, v_expect):
            np.testing.assert_almost_equal(sum([x ** 2 for x in v_expect]) ** 0.5, 1.0, decimal=2)
            np.testing.assert_almost_equal(cf.calc_rv(v_i, v_j), v_expect, decimal=3)

        test([0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 1.0])
        test([0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 1.0])
        test([0.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 0.707, 0.707])
        test([0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.447, 0.894])
        test([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0.577, 0.577, 0.577])
        test([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.267, 0.535, 0.802])
        test([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.267, 0.535, 0.802])
        test([-1.0, -2.0, -3.0], [-2.0, -4.0, -6.0], [-0.267, -0.535, -0.802])

    def test_calc_fv_ij(self):
        def test(mx_r_curr, arr_m_curr, i, j, expect_v):
            result_v = cf.calc_Fv_ij(i, j, mx_r_curr, arr_m_curr)
            numpy.testing.assert_almost_equal(result_v, expect_v, decimal=3)

        def test_with_coef(mx_r_curr, arr_m_curr, i, j, f_coef, expect_v):
            result_v = cf.calc_Fv_ij(i, j, mx_r_curr, arr_m_curr, f_coef=f_coef)
            numpy.testing.assert_almost_equal(result_v, expect_v, decimal=3)

        #   given
        mx_r = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]
        arr_m = [1.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1, 1.0, [0.0, 0.0, 1.0])
        test_with_coef(mx_r, arr_m, 0, 2, 1.0, [0.0, 0.25, 0.0])
        test_with_coef(mx_r, arr_m, 0, 3, 1.0, [0.111, 0.0, 0.0])
        test(mx_r, arr_m, 0, 1, [0.0, 0.0, 10.0])
        test(mx_r, arr_m, 0, 2, [0.0, 2.5, 0.0])
        test(mx_r, arr_m, 0, 3, [1.11, 0.0, 0.0])

        #   given
        mx_r = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1, 1.0, [0.0, 0.0, 2.0])
        test_with_coef(mx_r, arr_m, 0, 2, 1.0, [0.0, 0.5, 0.0])
        test_with_coef(mx_r, arr_m, 0, 3, 1.0, [0.222, 0.0, 0.0])
        test(mx_r, arr_m, 0, 1, [0.0, 0.0, 20.0])
        test(mx_r, arr_m, 0, 2, [0.0, 5.0, 0.0])
        test(mx_r, arr_m, 0, 3, [2.222, 0.0, 0.0])

    def test_calc_fv_sum(self):
        def test_with_coef(mx_r_curr, arr_m_curr, i, f_coef, expect_v):
            result_v = cf.calc_Fv_sum(i, mx_r_curr, arr_m_curr, f_coef=f_coef)
            numpy.testing.assert_almost_equal(result_v, expect_v, decimal=3)

        #   given
        mx_r = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1.0, [0.2222, 0.50, 2.0])

        #   given
        mx_r = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1.0, [0.0, 0.0, 0.0])

        #   given
        mx_r = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1.0, [-1.5, 0.0, 0.0])

        #   given
        mx_r = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1.0, [2.0, 0.0, 0.0])

        #   given
        mx_r = [[0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]]
        arr_m = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #   when then
        test_with_coef(mx_r, arr_m, 0, 1.0, [0.0, 0.0, 0.0])
