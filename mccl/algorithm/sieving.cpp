#include <mccl/algorithm/sieving.hpp>

MCCL_BEGIN_NAMESPACE

sieving_config_t sieving_config_default;

size_t intersection_elements(const element_t& x, const element_t& y, size_t element_weight)
{
	unsigned xi = 0, yi = 0, c = 0;
	while (true)
	{
		// xi < element_weight AND yi < element_weight
		if (x.first[xi] == y.first[yi])
		{
			++c; ++xi; ++yi;
			if (xi == element_weight || yi == element_weight)
				return c;
			continue;
		}
		if (x.first[xi] < y.first[yi])
		{
			++xi;
			if (xi == element_weight)
				return c;
			continue;
		}
		// (x.first[xi] > y.first[yi])
		++yi;
		if (yi == element_weight)
			return c;
	}
}

bool combine_elements(const element_t& x, const element_t& y, element_t& dest, size_t element_weight)
{
	unsigned xi = 0, yi = 0, di = 0;
	while (true)
	{
		if (xi >= element_weight)
		{
			if (yi != di)
				return false;
			for (; yi < element_weight; ++yi, ++di)
				dest.first[di] = y.first[yi];
			dest.second = x.second ^ y.second;
			return true;
		}
		if (yi >= element_weight)
		{
			if (xi != di)
				return false;
			for (; xi < element_weight; ++xi, ++di)
				dest.first[di] = x.first[xi];
			dest.second = x.second ^ y.second;
			return true;
		}
		if (x.first[xi] == y.first[yi])
		{
			++xi; ++yi;
			continue;
		}
		if (x.first[xi] < y.first[yi])
		{
			if (di == element_weight)
				return false;
			dest.first[di] = x.first[xi];
			++xi; ++di;
			continue;
		}
		// x.first[xi] > y.first[yi]
		if (di == element_weight)
			return false;
		dest.first[di] = y.first[yi];
		++yi; ++di;
	}
}

void combine_elements_v2(const element_t& x, const element_t& y, element_t& dest, size_t element_weight)
{
	unsigned xi = 0, yi = 0, di = 0;
	while (true)
	{
		if (xi >= element_weight)
		{
			for (; yi < element_weight; ++yi, ++di)
				dest.first[di] = y.first[yi];
			dest.second = x.second ^ y.second;
			return;
		}
		if (yi >= element_weight)
		{
			for (; xi < element_weight; ++xi, ++di)
				dest.first[di] = x.first[xi];
			dest.second = x.second ^ y.second;
			return;
		}
		if (x.first[xi] == y.first[yi])
		{
			++xi; ++yi;
			continue;
		}
		if (x.first[xi] < y.first[yi])
		{
			
			dest.first[di] = x.first[xi];
			++xi; ++di;
			continue;
		}
		else// x.first[xi] > y.first[yi]
		{
			dest.first[di] = y.first[yi];
			++yi; ++di;
		}
	}
}

size_t binomial_coeff(size_t n, size_t k)
{
	if (k > n)
		return 0;
	if (k == 0 || k == n)
		return 1;

	return binomial_coeff(n - 1, k - 1)
		+ binomial_coeff(n - 1, k);
}


vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_sieving subISDT;
    ISD_sieving<> ISD(subISDT);

    return solve_SD(ISD, H, S, w);
}

vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_sieving subISDT;
    ISD_sieving<> ISD(subISDT);

    subISDT.load_config(configmap);
    ISD.load_config(configmap);

    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE