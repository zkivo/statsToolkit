import unittest
import numpy as np
from scipy import stats as st
from ..probability import *


class TestProbabilityFunctions(unittest.TestCase):

    def test_binopdf(self):
        """Test binopdf function."""
        result = binopdf(3, 10, 0.5)
        expected = st.binom.pmf(3, 10, 0.5)
        self.assertAlmostEqual(result, expected)

    def test_poisspdf(self):
        """Test poisspdf function."""
        result = poisspdf(2, 5)
        expected = st.poisson.pmf(2, 5)
        self.assertAlmostEqual(result, expected)

    def test_geopdf(self):
        """Test geopdf function."""
        result = geopdf(2, 0.3)
        expected = st.geom.pmf(2, 0.3)
        self.assertAlmostEqual(result, expected)

    def test_nbinpdf(self):
        """Test nbinpdf function."""
        result = nbinpdf(3, 5, 0.5)
        expected = st.nbinom.pmf(3, 5, 0.5)
        self.assertAlmostEqual(result, expected)

    def test_hygepdf(self):
        """Test hygepdf function."""
        result = hygepdf(2, 30, 10, 5)
        expected = st.hypergeom.pmf(2, 30, 10, 5)
        self.assertAlmostEqual(result, expected)

    def test_betapdf(self):
        """Test betapdf function."""
        result = betapdf(0.5, 2, 5)
        expected = st.beta.pdf(0.5, 2, 5)
        self.assertAlmostEqual(result, expected)

    def test_chi2pdf(self):
        """Test chi2pdf function."""
        result = chi2pdf(3, 2)
        expected = st.chi2.pdf(3, 2)
        self.assertAlmostEqual(result, expected)

    def test_exppdf(self):
        result = exppdf(2, scale=3)
        expected = st.expon.pdf(2, scale=3)
        self.assertAlmostEqual(result, expected)

    def test_fpdf(self):
        """Test fpdf function."""
        result = fpdf(1, 5, 2)
        expected = st.f.pdf(1, 5, 2)
        self.assertAlmostEqual(result, expected)

    def test_normpdf(self):
        """Test normpdf function."""
        result = normpdf(1, 0, 1)
        expected = st.norm.pdf(1, 0, 1)
        self.assertAlmostEqual(result, expected)

    def test_lognpdf(self):
        result = lognpdf(2, s=0.5, scale=1)
        expected = st.lognorm.pdf(2, s=0.5, scale=1)
        self.assertAlmostEqual(result, expected)

    def test_tpdf(self):
        """Test tpdf function."""
        result = tpdf(0.5, 10)
        expected = st.t.pdf(0.5, 10)
        self.assertAlmostEqual(result, expected)

    def test_wblpdf(self):
        """Test wblpdf function."""
        result = wblpdf(1.5, 2)
        expected = st.weibull_min.pdf(1.5, 2)
        self.assertAlmostEqual(result, expected)

    def test_mvnpdf(self):
        """Test mvnpdf function."""
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        result = mvnpdf([0, 0], mean, cov)
        expected = st.multivariate_normal.pdf([0, 0], mean, cov)
        self.assertAlmostEqual(result, expected)

    # Similarly for CDF tests
    def test_binocdf(self):
        """Test binocdf function."""
        result = binocdf(3, 10, 0.5)
        expected = st.binom.cdf(3, 10, 0.5)
        self.assertAlmostEqual(result, expected)

    def test_poisscdf(self):
        """Test poisscdf function."""
        result = poisscdf(2, 5)
        expected = st.poisson.cdf(2, 5)
        self.assertAlmostEqual(result, expected)

    def test_geocdf(self):
        """Test geocdf function."""
        result = geocdf(2, 0.3)
        expected = st.geom.cdf(2, 0.3)
        self.assertAlmostEqual(result, expected)

    def test_nbincdf(self):
        """Test nbincdf function."""
        result = nbincdf(3, 5, 0.5)
        expected = st.nbinom.cdf(3, 5, 0.5)
        self.assertAlmostEqual(result, expected)

    def test_hygecdf(self):
        """Test hygecdf function."""
        result = hygecdf(2, 30, 10, 5)
        expected = st.hypergeom.cdf(2, 30, 10, 5)
        self.assertAlmostEqual(result, expected)

    def test_betacdf(self):
        """Test betacdf function."""
        result = betacdf(0.5, 2, 5)
        expected = st.beta.cdf(0.5, 2, 5)
        self.assertAlmostEqual(result, expected)

    def test_chi2cdf(self):
        """Test chi2cdf function."""
        result = chi2cdf(3, 2)
        expected = st.chi2.cdf(3, 2)
        self.assertAlmostEqual(result, expected)

    def test_expcdf(self):
        result = expcdf(2, scale=3)
        expected = st.expon.cdf(2, scale=3)
        self.assertAlmostEqual(result, expected)

    def test_fcdf(self):
        """Test fcdf function."""
        result = fcdf(1, 5, 2)
        expected = st.f.cdf(1, 5, 2)
        self.assertAlmostEqual(result, expected)

    def test_normcdf(self):
        """Test normcdf function."""
        result = normcdf(1, 0, 1)
        expected = st.norm.cdf(1, 0, 1)
        self.assertAlmostEqual(result, expected)

    def test_logncdf(self):
        result = logncdf(2, s=0.5, scale=1)
        expected = st.lognorm.cdf(2, s=0.5, scale=1)
        self.assertAlmostEqual(result, expected)

    def test_tcdf(self):
        """Test tcdf function."""
        result = tcdf(0.5, 10)
        expected = st.t.cdf(0.5, 10)
        self.assertAlmostEqual(result, expected)

    def test_wblcdf(self):
        """Test wblcdf function."""
        result = wblcdf(1.5, 2)
        expected = st.weibull_min.cdf(1.5, 2)
        self.assertAlmostEqual(result, expected)

    # Percent point function (Inverse of CDF)
    def test_norminv(self):
        """Test norminv function."""
        result = norminv(0.95, 0, 1)
        expected = st.norm.ppf(0.95, 0, 1)
        self.assertAlmostEqual(result, expected)

    def test_tinv(self):
        """Test tinv function."""
        result = tinv(0.95, 10)
        expected = st.t.ppf(0.95, 10)
        self.assertAlmostEqual(result, expected)

    def test_chi2inv(self):
        """Test chi2inv function."""
        result = chi2inv(0.95, 2)
        expected = st.chi2.ppf(0.95, 2)
        self.assertAlmostEqual(result, expected)

    def test_finv(self):
        """Test finv function."""
        result = finv(0.95, 5, 2)
        expected = st.f.ppf(0.95, 5, 2)
        self.assertAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
