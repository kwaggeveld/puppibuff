-- Written by `puppibuff`. 
-- End to end: **total** clock cycles over **n_intervals** integration steps.

library ieee;
use ieee.std_logic_1164.ALL;
use ieee.numeric_std.all;

use work.ipbus.all;
use work.emp_data_types.all;
use work.emp_project_decl.all;

use work.emp_device_decl;
use work.emp_ttc_decl.all;
use work.emp_slink_types.all;

use work.tcds2_streams_pkg;

entity emp_payload is
  port(
    clk         : in  std_logic;        -- ipbus signals
    rst         : in  std_logic;
    ipb_in      : in  ipb_wbus;
    ipb_out     : out ipb_rbus;
    clk40       : in  std_logic;
    clk_payload : in  std_logic_vector(2 downto 0);
    rst_payload : in  std_logic_vector(2 downto 0);
    clk_p       : in  std_logic;        -- data clock
    rst_loc     : in  std_logic_vector(emp_device_decl.N_REGION - 1 downto 0);
    clken_loc   : in  std_logic_vector(emp_device_decl.N_REGION - 1 downto 0);
    ctrs        : in  ttc_stuff_array;
    ttc2        : in  tcds2_streams_pkg.tcds2_ttc2;
    d           : in  ldata(4 * emp_device_decl.N_REGION - 1 downto 0);  -- data in
    q           : out ldata(4 * emp_device_decl.N_REGION - 1 downto 0);  -- data out
    gpio        : out std_logic_vector(29 downto 0);  -- IO to mezzanine connector
    gpio_en     : out std_logic_vector(29 downto 0);  -- IO to mezzanine connector (three-state enables)
    slink_q     : out slink_input_data_quad_array(SLINK_MAX_QUADS-1 downto 0);
    tts_busy    : out std_logic;
    backpressure : in std_logic_vector(SLINK_MAX_QUADS-1 downto 0);
    -- External electrical signals (e.g. Apollo HDMI, Serenity HDR, RTM Zone 3)
    ext_lvds_in  : in  std_logic_vector(emp_device_decl.N_LVDS_IN - 1 downto 0);
    ext_lvds_out : out std_logic_vector(emp_device_decl.N_LVDS_OUT - 1 downto 0)
    );
end emp_payload;

architecture rtl of emp_payload is

  -- Extracted from each block's csynth report
  constant AB2_LATENCY   : integer := **ab2_latency**;
  constant FIELD_LATENCY : integer_vector(0 to **last_interval**) := (**field_table**);

  -- How long a velocity is held to become the next step's v_prev
  constant HOLD : integer_vector(0 to **last_interval**) := (**hold_table**);

  type state_arr1d   is array (natural range <>) of std_logic_vector(**state_msb** downto 0);
  type accum_arr1d   is array (natural range <>) of std_logic_vector(**accum_msb** downto 0);
  type decoded_arr1d is array (natural range <>) of std_logic_vector(**decoded_msb** downto 0);

  type state_arr2d is array (natural range <>) of state_arr1d;
  type accum_arr2d is array (natural range <>) of accum_arr1d;

  -- x<s> enters step <s>, s<s>_d is its narrowed copy, 
  -- s<s>_q the velocity at that step predicted by the BDTs.
**declarations**
  signal x_out   : accum_arr1d(**channels**);
  signal decoded : decoded_arr1d(0 to **last_decoded**);

begin

  -- emp-fwk generic things, tying off various unused ports
  ipb_out <= IPB_RBUS_NULL;
  gpio    <= (others => '0');
  gpio_en <= (others => '0');

  -- Noise arrives on **n_channels** links from **in_base**
**inputs**

  -- The **n_decoded** values of one sample leave on links from **out_base**
**outputs**

  -- Hold every signal until the block that reads it catches up
  pipe_proc : process(clk_p)
  begin
    if rising_edge(clk_p) then
**pipes**
    end if;
  end process;

  -- Narrow accum_t -> state_t, dropping the low **dropped_bits** fractional bits
  GenNarrow:
  for idx in **channels** generate
**narrow**
  end generate;

**instances**  --------------------------------------------------------------------------
  -- Decoding

**decode**
end rtl;
