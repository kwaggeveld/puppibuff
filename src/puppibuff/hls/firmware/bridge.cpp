#include "firmware/ap_types.h"
#include "firmware/flowhls.h"

#include <vector>
#include <stdexcept>


std::vector<double> field(size_t step, std::vector<double> const &x)
{
    size_t const n_events = x.size() / n_channels;
    std::vector<double> v(x.size());

    for (size_t event = 0; event != n_events; ++event)
    {
        state_arr_t xt, vt;
                                        // Cast input to state_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<state_t>(x[event * n_channels + idx]);

        switch (step)
        {
**cases**
            default: throw std::out_of_range("No such step");
        }
                                        // Cast output to double
        for (size_t idx = 0; idx != n_channels; ++idx)
            v[event * n_channels + idx] = static_cast<double>(vt[idx]);
    }

    return v;
}

std::vector<double> sample(std::vector<double> const &x0)
{
    size_t const n_events = x0.size() / n_channels;
    std::vector<double> x1(x0.size());

    for (size_t event = 0; event != n_events; ++event)
    {
        accum_arr_t xt, yt;
                                        // Cast noise to accum_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<accum_t>(x0[event * n_channels + idx]);

        **sample_top**(xt, yt);
                                        // Cast sample to double
        for (size_t idx = 0; idx != n_channels; ++idx)
            x1[event * n_channels + idx] = static_cast<double>(yt[idx]);
    }

    return x1;
}


std::vector<double> decode_batch(std::vector<double> const &x)
{
    size_t const n_events = x.size() / n_channels;
    std::vector<double> out(n_events * n_decoded);

    for (size_t event = 0; event != n_events; ++event)
    {
        accum_arr_t xt;
        decoded_arr_t decoded;
                                        // Cast sample to accum_t
        for (size_t idx = 0; idx != n_channels; ++idx)
            xt[idx] = static_cast<accum_t>(x[event * n_channels + idx]);

        **decode_top**(xt, decoded);
                                        // Cast physical output to double
        for (size_t idx = 0; idx != n_decoded; ++idx)
            out[event * n_decoded + idx] = static_cast<double>(decoded[idx]);
    }

    return out;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(conifer_bridge_**bridge_module**, m)
{
    m.def("field",  &field);
    m.def("sample", &sample);
    m.def("decode", &decode_batch);
}

