import QuantLib as ql
import numpy as np

class SyntheticSwapEngine:
    def __init__(self, base_rate: float, a: float, sigma: float, notional: float, maturity_years: int):
        """
        Initializes the QuantLib pricing templates and models.
        All inputs are purely synthetic. No real-world data files allowed.
        """
        self.base_rate = base_rate
        self.a = a
        self.sigma = sigma
        self.notional = notional
        self.maturity_years = maturity_years

        # Set up reference date
        self.today = ql.Date(12, ql.June, 2026)
        ql.Settings.instance().evaluationDate = self.today

        # Calendars and day counters
        self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        settlement_days = 2
        self.settlement_date = self.calendar.advance(self.today, settlement_days, ql.Days)
        self.day_count = ql.Actual360()

        # 1. Initial Flat Curve (fixed, used by the Model)
        flat_curve = ql.FlatForward(self.today, base_rate, self.day_count)
        self.model_curve_handle = ql.YieldTermStructureHandle(flat_curve)

        # 2. Relinkable Curve Handle (used by the Pricing Engine and Index)
        self.pricing_curve_handle = ql.RelinkableYieldTermStructureHandle(flat_curve)

        # Floating Index
        self.index = ql.USDLibor(ql.Period(6, ql.Months), self.pricing_curve_handle)

        # Swap specifications
        self.maturity_date = self.calendar.advance(self.settlement_date, maturity_years, ql.Years)
        
        fixed_schedule = ql.Schedule(
            self.settlement_date, self.maturity_date, ql.Period(ql.Annual), self.calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False
        )
        
        floating_schedule = ql.Schedule(
            self.settlement_date, self.maturity_date, ql.Period(6, ql.Months), self.calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False
        )

        fixed_day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        floating_day_count = ql.Actual360()

        # Determine the fair swap rate at inception (t=0)
        dummy_swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer, notional, fixed_schedule, base_rate, fixed_day_count,
            floating_schedule, self.index, 0.0, floating_day_count
        )
        engine = ql.DiscountingSwapEngine(self.pricing_curve_handle)
        dummy_swap.setPricingEngine(engine)
        self.fair_rate = dummy_swap.fairRate()

        # Re-instantiate swap with the fair rate
        self.swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer, notional, fixed_schedule, self.fair_rate, fixed_day_count,
            floating_schedule, self.index, 0.0, floating_day_count
        )
        self.swap.setPricingEngine(engine)

        # Add historical fixing for the first coupon period to avoid missing fixing errors
        fixing_date = self.index.fixingDate(self.settlement_date)
        # Note: QuantLib requires index.fixing(fixing_date) to fetch the fixing rate from the curve
        self.index.addFixing(fixing_date, self.index.fixing(fixing_date))

        # Instantiate Hull-White model
        self.model = ql.HullWhite(self.model_curve_handle, a, sigma)

    def generate_trajectory_batch(self, num_paths: int, num_days: int, seed: int = 42) -> dict:
        """
        Executes the batch stochastic simulation and pathwise pricing loop.
        
        Returns a dictionary containing:
        - "time_grid": Shape (num_days,) containing t values.
        - "short_rates": Shape (num_paths, num_days) containing simulated r_t values.
        - "mtm_profiles": Shape (num_paths, num_days) containing the swap valuations (V_t).
        - "exposure_profiles": Shape (num_paths, num_days) containing max(V_t, 0).
        """
        dt = 1.0 / 365.0
        time_grid = np.arange(num_days) * dt

        # Vectorized HW1F path simulation
        np.random.seed(seed)
        
        # We start with x_0 = 0, which gives r_0 = base_rate
        x = np.zeros((num_paths, num_days))
        if num_days > 1:
            Z = np.random.normal(size=(num_paths, num_days - 1))
            factor = np.exp(-self.a * dt)
            std_dev = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.a * dt)) / (2.0 * self.a))
            for i in range(1, num_days):
                x[:, i] = x[:, i-1] * factor + std_dev * Z[:, i-1]

        # Shift x by analytical alpha(t) to fit the initial flat term structure
        alpha = self.base_rate + (self.sigma**2 / (2.0 * self.a**2)) * (1.0 - np.exp(-self.a * time_grid))**2
        short_rates = x + alpha

        # Allocate arrays for profiles
        mtm_profiles = np.zeros((num_paths, num_days))
        exposure_profiles = np.zeros((num_paths, num_days))

        # Prepare dates for loop
        eval_dates = [self.today + i for i in range(num_days)]
        
        # Grid offsets for discount curve (semi-annual up to maturity_years + 1)
        grid_offsets = np.arange(0.0, self.maturity_years + 1.5, 0.5)

        # Loop daily (outer) to minimize evaluationDate notifications in QuantLib
        for d_idx in range(num_days):
            eval_date = eval_dates[d_idx]
            ql.Settings.instance().evaluationDate = eval_date
            
            t_val = self.day_count.yearFraction(self.today, eval_date)

            # Pre-calculate future dates for the reconstructed discount curve
            curve_dates = [eval_date]
            for offset in grid_offsets[1:]:
                # We can approximate or advance accurately by calendar
                # 6 months is approx 182 days, let's use calendar days for speed and simplicity
                days_offset = int(offset * 365.25)
                curve_dates.append(eval_date + days_offset)

            # Compute year fractions for these grid dates from the simulation start date
            T_j_list = [self.day_count.yearFraction(self.today, d) for d in curve_dates]

            # Pathwise loop
            for p_idx in range(num_paths):
                r_sim = short_rates[p_idx, d_idx]

                # Compute analytical zero-coupon bond prices from the HW1F model
                discount_factors = [1.0]
                for T_j in T_j_list[1:]:
                    df = self.model.discountBond(t_val, T_j, r_sim)
                    discount_factors.append(df)

                # Construct the discount curve at eval_date
                new_curve = ql.DiscountCurve(curve_dates, discount_factors, self.day_count, self.calendar)
                self.pricing_curve_handle.linkTo(new_curve)

                # Price the swap
                npv = self.swap.NPV()
                mtm_profiles[p_idx, d_idx] = npv
                exposure_profiles[p_idx, d_idx] = max(npv, 0.0)

        # Reset evaluation date to today when finished to avoid side effects
        ql.Settings.instance().evaluationDate = self.today

        return {
            "time_grid": time_grid,
            "short_rates": short_rates,
            "mtm_profiles": mtm_profiles,
            "exposure_profiles": exposure_profiles
        }
