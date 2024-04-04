#include <mccl/algorithm/sieving.hpp>

MCCL_BEGIN_NAMESPACE

sieving_config_t sieving_config_default;
size_t loop_it = 0;

size_t intersection_elements(const element_t& x, const element_t& y, size_t w)
{
	unsigned xi = 0, yi = 0, c = 0;
	while (true)
	{
		// xi < element_weight AND yi < element_weight
		if (x.first[xi] == y.first[yi])
		{
			++c; ++xi; ++yi;
			if (xi == w || yi == w)
				return c;
			continue;
		}
		if (x.first[xi] < y.first[yi])
		{
			++xi;
			if (xi == w)
				return c;
			continue;
		}
		// (x.first[xi] > y.first[yi])
		++yi;
		if (yi == w)
			return c;
	}
}

size_t intersection_elements(const element_t& x, const center_t& y, size_t x_w, size_t y_w)
{
	unsigned xi = 0, yi = 0, c = 0;
	while (true)
	{
		// xi < element_weight AND yi < element_weight
		if (x.first[xi] == y.first[yi])
		{
			++c; ++xi; ++yi;
			if (xi == x_w || yi == y_w)
				return c;
			continue;
		}
		if (x.first[xi] < y.first[yi])
		{
			++xi;
			if (xi == x_w)
				return c;
			continue;
		}
		// (x.first[xi] > y.first[yi])
		++yi;
		if (yi == y_w)
			return c;
	}
}

bool combine_elements(const element_t& x, const element_t& y, element_t& dest, size_t w)
{
	unsigned xi = 0, yi = 0, di = 0;
	while (true)
	{
		if (xi >= w)
		{
			if (yi != di)
				return false;
			for (; yi < w; ++yi, ++di)
				dest.first[di] = y.first[yi];
			dest.second = x.second ^ y.second;
			return true;
		}
		if (yi >= w)
		{
			if (xi != di)
				return false;
			for (; xi < w; ++xi, ++di)
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
			if (di == w)
				return false;
			dest.first[di] = x.first[xi];
			++xi; ++di;
			continue;
		}
		// x.first[xi] > y.first[yi]
		if (di == w)
			return false;
		dest.first[di] = y.first[yi];
		++yi; ++di;
	}
}

void sample_vec(size_t element_weight, size_t rows, size_t output_length, const std::vector<uint64_t>& firstwords, mccl_base_random_generator rnd, database& output)
{
	output.clear();

	uint64_t rnd_val;
	element_t element;

	for (auto& i : element.first)
	{
		i = 0; i = ~i; // set all row indices to invalid positions
	}

	while (output.size() < output_length)
	{
		element.second = 0;
		unsigned k = 0;
		while (k < element_weight)
		{
			element.first[k] = rnd() % rows;
			// try both pieces of code
#if 0
				// I think this obtains the same i as the code below, but with binary search and with a single line of code
			unsigned i = std::lower_bound(element.first.begin(), element.first.begin() + k) - element.first.begin();
#else
				// I think this is correct, but linear search
			unsigned i = k;
			while (i > 0)
			{
				if (element.first[i - 1] < element.first[k])
					break;
				--i;
			}
#endif
			// PROPERTY: i is largest i such that (i==0) OR (element.first[i-1] < element.first[k])
			// that means is the smallest i such that element.first[i] >= element.first[k] (otherwise i should be at least 1 larger)
			// if element.first[i] == element.first[k] then we sample the same index twice and we need to resample element.first[k]
			if (i < k && element.first[i] == element.first[k])
				continue;
			// update value
			element.second ^= firstwords[element.first[k]];
			// now move k at position i
			if (i < k)
			{
				auto firstk = element.first[k];
				for (unsigned j = k; j > i; --j)
					element.first[j] = element.first[j - 1];
				element.first[i] = firstk;
			}
			++k;
		}
		// already sorted and unique indices now
#if 0            
			// sort indices
		std::sort(element.first.begin(), element.first.begin() + element_weight);
		// if there are any double occurences they appear next to each other
		bool ok = true;
		for (unsigned k = 1; k < element_weight; ++k)
			if (element.first[k - 1] == element.first[k])
			{
				ok = false;
				break;
			}
		// if there are double occurences we resample element
		if (!ok)
			continue;
#endif
		output.insert(element);
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