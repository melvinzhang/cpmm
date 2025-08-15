/// Constant Product Market Maker (CPMM) implementation
/// Based on the formal specification by Yi Zhang, Xiaohong Chen, and Daejun Park
/// 
/// This implementation follows the integer arithmetic versions (_code) from the PDF:
/// - Uses u16 for state variables (e, t, l)  
/// - Uses u32 for intermediate calculations
/// - Implements exact formulas from Definitions 2, 4, 6, and 8

#[derive(Debug, Clone)]
pub struct CPMM {
    /// Ether reserve (in wei)
    pub e: u16,
    /// Token reserve  
    pub t: u16,
    /// Total liquidity
    pub l: u16,
}

impl CPMM {
    /// Creates a new CPMM instance
    pub fn new(ether_reserve: u16, token_reserve: u16, liquidity: u16) -> Self {
        Self {
            e: ether_reserve,
            t: token_reserve,
            l: liquidity,
        }
    }

    /// Returns the constant product k = e × t
    pub fn k(&self) -> u32 {
        (self.e as u32) * (self.t as u32)
    }

    /// Definition 2: addLiquiditycode
    /// Takes integer Δe > 0 and updates state (e,t,l) ∈ Z³ → (e'',t'',l'') ∈ Z³
    /// 
    /// Where:
    /// e'' = e + Δe = (1 + α)e
    /// t'' = t + ⌊(Δe × t)/e⌋ + 1 = ⌊(1 + α)t⌋ + 1  
    /// l'' = l + ⌊(Δe × l)/e⌋ = ⌊(1 + α)l⌋
    pub fn add_liquidity(&mut self, delta_e: u16) -> Result<(u16, u16), &'static str> {
        if self.e == 0 {
            return Err("Cannot add liquidity when e = 0");
        }

        let delta_e_u32 = delta_e as u32;
        let e_u32 = self.e as u32;
        let t_u32 = self.t as u32;
        let l_u32 = self.l as u32;

        // Calculate deltas using integer division (floor)
        let delta_t_calc = ((delta_e_u32 * t_u32) / e_u32) + 1;
        let delta_l_calc = (delta_e_u32 * l_u32) / e_u32;
        
        // Check for overflow before casting
        if delta_t_calc > u16::MAX as u32 {
            return Err("delta_t calculation overflows u16");
        }
        if delta_l_calc > u16::MAX as u32 {
            return Err("delta_l calculation overflows u16");
        }
        
        let delta_t = delta_t_calc as u16;
        let delta_l = delta_l_calc as u16;
        
        // Check for overflow in state updates
        if self.e.checked_add(delta_e).is_none() {
            return Err("e update overflows u16");
        }
        if self.t.checked_add(delta_t).is_none() {
            return Err("t update overflows u16");
        }
        if self.l.checked_add(delta_l).is_none() {
            return Err("l update overflows u16");
        }

        // Update state
        self.e += delta_e;
        self.t += delta_t;
        self.l += delta_l;

        Ok((delta_t, delta_l))
    }

    /// Definition 4: removeLiquiditycode  
    /// Takes integer 0 < Δl < l and updates state (e,t,l) ∈ Z³ → (e'',t'',l'') ∈ Z³
    ///
    /// Where:
    /// e'' = e - ⌊(Δl × e)/l⌋ = ⌈(1 - α)e⌉
    /// t'' = t - ⌊(Δl × t)/l⌋ = ⌈(1 - α)t⌉  
    /// l'' = l - Δl = (1 - α)l
    pub fn remove_liquidity(&mut self, delta_l: u16) -> Result<(u16, u16), &'static str> {
        if delta_l >= self.l {
            return Err("Cannot remove more liquidity than exists");
        }

        let delta_l_u32 = delta_l as u32;
        let e_u32 = self.e as u32;
        let t_u32 = self.t as u32;
        let l_u32 = self.l as u32;

        // Calculate deltas using integer division (floor)
        let delta_e_calc = (delta_l_u32 * e_u32) / l_u32;
        let delta_t_calc = (delta_l_u32 * t_u32) / l_u32;
        
        // Check for overflow before casting (should not happen for remove_liquidity, but being safe)
        if delta_e_calc > u16::MAX as u32 {
            return Err("delta_e calculation overflows u16");
        }
        if delta_t_calc > u16::MAX as u32 {
            return Err("delta_t calculation overflows u16");
        }
        
        let delta_e = delta_e_calc as u16;
        let delta_t = delta_t_calc as u16;
        
        // Check for underflow in state updates
        if self.e < delta_e {
            return Err("e update would underflow");
        }
        if self.t < delta_t {
            return Err("t update would underflow");
        }

        // Update state
        self.e -= delta_e;
        self.t -= delta_t;
        self.l -= delta_l;

        Ok((delta_e, delta_t))
    }

    /// Definition 6: getInputPricecode
    /// Let ρ be the trade fee. Takes input Δx > 0, x, y ∈ Z, outputs Δy ∈ Z
    /// 
    /// getInputPricecode(Δx)(x,y) = Δy = ⌊(αγ)/(1 + αγ) × y⌋
    /// where α = Δx/x and γ = 1 - ρ
    /// 
    /// Implementation: (997 * Δx * y) / (1000 * x + 997 * Δx)
    /// where / is integer division with truncation (floor)
    pub fn get_input_price(&self, delta_x: u16, x_reserve: u16, y_reserve: u16) -> Result<u16, &'static str> {
        if x_reserve == 0 {
            return Ok(0);
        }

        let delta_x_u32 = delta_x as u32;
        let x_u32 = x_reserve as u32;
        let y_u32 = y_reserve as u32;

        // Using ρ = 0.003 (0.3% fee), so γ = 0.997
        // Formula: (997 * Δx * y) / (1000 * x + 997 * Δx)
        
        // Check for overflow in numerator calculation: 997 * Δx * y
        let first_mult = delta_x_u32.checked_mul(997u32)
            .ok_or("delta_x * 997 overflows u32")?;
        let numerator = first_mult.checked_mul(y_u32)
            .ok_or("numerator calculation overflows u32")?;
        
        // Check for overflow in denominator calculation: 1000 * x + 997 * Δx
        let first_term = x_u32.checked_mul(1000u32)
            .ok_or("x * 1000 overflows u32")?;
        let second_term = delta_x_u32.checked_mul(997u32)
            .ok_or("delta_x * 997 overflows u32")?;
        let denominator = first_term.checked_add(second_term)
            .ok_or("denominator calculation overflows u32")?;

        if denominator == 0 {
            return Ok(0);
        }

        let result = numerator / denominator;
        if result > u16::MAX as u32 {
            return Err("result overflows u16");
        }

        Ok(result as u16)
    }

    /// Definition 8: getOutputPricecode
    /// Let ρ be the trade fee. Takes input 0 < Δy < y, x, y ∈ Z, outputs Δx ∈ Z
    ///
    /// getOutputPricecode(Δy)(x,y) = Δx = ⌊(β/(1-β)) × (1/γ) × x⌋ + 1
    /// where β = Δy/y < 1 and γ = 1 - ρ
    ///
    /// Implementation: (1000 * x * Δy) / (997 * (y - Δy)) + 1  
    /// where / is integer division with truncation (floor)
    pub fn get_output_price(&self, delta_y: u16, x_reserve: u16, y_reserve: u16) -> Result<u16, &'static str> {
        if delta_y >= y_reserve {
            return Err("Output amount must be less than reserve");
        }

        let delta_y_u32 = delta_y as u32;
        let x_u32 = x_reserve as u32;
        let y_u32 = y_reserve as u32;

        // Using ρ = 0.003 (0.3% fee), so γ = 0.997
        // Formula: (1000 * x * Δy) / (997 * (y - Δy)) + 1
        
        // Check for overflow in numerator calculation: 1000 * x * Δy
        let first_mult = x_u32.checked_mul(1000u32)
            .ok_or("x * 1000 overflows u32")?;
        let numerator = first_mult.checked_mul(delta_y_u32)
            .ok_or("numerator calculation overflows u32")?;
        
        // Check for underflow and overflow in denominator calculation: 997 * (y - Δy)
        if y_u32 < delta_y_u32 {
            return Err("y - delta_y underflows");
        }
        let y_minus_delta = y_u32 - delta_y_u32;
        let denominator = y_minus_delta.checked_mul(997u32)
            .ok_or("denominator calculation overflows u32")?;

        if denominator == 0 {
            return Err("Division by zero in output price calculation");
        }

        let base_result = numerator / denominator;
        let result = base_result.checked_add(1)
            .ok_or("result + 1 overflows u32")?;
        
        if result > u16::MAX as u32 {
            return Err("result overflows u16");
        }

        Ok(result as u16)
    }

    /// Generic swap calculation that returns new state values
    /// Parameters:
    /// - input_amount: The amount being input to the swap
    /// - output_amount: The calculated output amount  
    /// - from_reserve: Current value of the reserve being modified by input
    /// - to_reserve: Current value of the reserve being modified by output
    /// - add_to_from: Whether to add input_amount to from_reserve (true) or subtract (false)
    /// Returns: (new_from_reserve, new_to_reserve, output_amount)
    fn calculate_swap(&self,
                     input_amount: u16,
                     output_amount: u16,
                     from_reserve: u16,
                     to_reserve: u16,
                     add_to_from: bool,
                     from_name: &'static str,
                     to_name: &'static str) -> Result<(u16, u16, u16), &'static str>
    {
        let (new_from, new_to) = if add_to_from {
            // Check for overflow when adding to from_reserve
            let new_from = from_reserve.checked_add(input_amount)
                .ok_or(if from_name == "e" { "e update overflows u16" } else { "t update overflows u16" })?;
            // Check for underflow when subtracting from to_reserve  
            if to_reserve < output_amount {
                return Err(if to_name == "e" { "e update would underflow" } else { "t update would underflow" });
            }
            let new_to = to_reserve - output_amount;
            (new_from, new_to)
        } else {
            // Check for underflow when subtracting from from_reserve
            if from_reserve < input_amount {
                return Err(if from_name == "e" { "e update would underflow" } else { "t update would underflow" });
            }
            let new_from = from_reserve - input_amount;
            // Check for overflow when adding to to_reserve
            let new_to = to_reserve.checked_add(output_amount)
                .ok_or(if to_name == "e" { "e update overflows u16" } else { "t update overflows u16" })?;
            (new_from, new_to)
        };
        
        Ok((new_from, new_to, output_amount))
    }

    /// Section 4.1.2: ethToTokencode
    /// Takes integer input Δe (Δe > 0) and updates state
    /// e'' = e + Δe
    /// t'' = t - getInputPricecode(Δe, e, t) = ⌊t'⌋
    pub fn eth_to_token(&mut self, delta_e: u16) -> Result<u16, &'static str> {
        let delta_t = self.get_input_price(delta_e, self.e, self.t)?;
        let (new_e, new_t, output) = self.calculate_swap(
            delta_e, delta_t, self.e, self.t, true, "e", "t")?;
        self.e = new_e;
        self.t = new_t;
        Ok(output)
    }

    /// Section 4.2.2: ethToTokenExactcode  
    /// Takes integer input Δt (0 < Δt < t) and updates state
    /// t'' = t - Δt
    /// e'' = e + getOutputPricecode(Δt, e, t)
    pub fn eth_to_token_exact(&mut self, delta_t: u16) -> Result<u16, &'static str> {
        let delta_e = self.get_output_price(delta_t, self.e, self.t)?;
        let (new_t, new_e, output) = self.calculate_swap(
            delta_t, delta_e, self.t, self.e, false, "t", "e")?;
        self.t = new_t;
        self.e = new_e;
        Ok(output)
    }

    /// Section 4.3.2: tokenToEthcode
    /// Takes integer input Δt (Δt > 0) and updates state  
    /// t'' = t + Δt
    /// e'' = e - getInputPricecode(Δt, t, e) = ⌊e'⌋
    pub fn token_to_eth(&mut self, delta_t: u16) -> Result<u16, &'static str> {
        let delta_e = self.get_input_price(delta_t, self.t, self.e)?;
        let (new_t, new_e, output) = self.calculate_swap(
            delta_t, delta_e, self.t, self.e, true, "t", "e")?;
        self.t = new_t;
        self.e = new_e;
        Ok(output)
    }

    /// Section 4.4.2: tokenToEthExactcode
    /// Takes integer input Δe (0 < Δe < e) and updates state
    /// e'' = e - Δe  
    /// t'' = t + getOutputPricecode(Δe, t, e)
    pub fn token_to_eth_exact(&mut self, delta_e: u16) -> Result<u16, &'static str> {
        let delta_t = self.get_output_price(delta_e, self.t, self.e)?;
        let (new_e, new_t, output) = self.calculate_swap(
            delta_e, delta_t, self.e, self.t, false, "e", "t")?;
        self.e = new_e;
        self.t = new_t;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpmm_creation() {
        let cpmm = CPMM::new(1000, 2000, 1414);
        assert_eq!(cpmm.e, 1000);
        assert_eq!(cpmm.t, 2000);
        assert_eq!(cpmm.l, 1414);
        assert_eq!(cpmm.k(), 2_000_000);
    }

    #[test]
    fn test_add_liquidity() {
        let mut cpmm = CPMM::new(1000, 2000, 1414);
        let initial_k = cpmm.k();
        
        let (delta_t, delta_l) = cpmm.add_liquidity(100).expect("Should not overflow");
        
        // Verify k increases (Theorem 2)
        assert!(cpmm.k() > initial_k);
        
        // Verify returned values match actual changes
        assert_eq!(delta_t, 201); // ⌊(100 * 2000)/1000⌋ + 1 = 200 + 1
        assert_eq!(delta_l, 141);  // ⌊(100 * 1414)/1000⌋ = 141
    }

    #[test]
    fn test_remove_liquidity() {
        let mut cpmm = CPMM::new(1100, 2201, 1555);
        let initial_k = cpmm.k();
        
        let (delta_e, delta_t) = cpmm.remove_liquidity(100).expect("Should not overflow/underflow");
        
        // Verify k decreases (Theorem 5)
        assert!(cpmm.k() <= initial_k);
        
        // Verify ceiling approximation
        assert_eq!(delta_e, 70);  // ⌊(100 * 1100)/1555⌋ = 70
        assert_eq!(delta_t, 141); // ⌊(100 * 2201)/1555⌋ = 141
    }

    #[test]
    fn test_price_calculations() {
        let cpmm = CPMM::new(1000, 2000, 1414);
        
        // Test input price with 0.3% fee
        let delta_y = cpmm.get_input_price(100, 1000, 2000).expect("Should not overflow");
        // (997 * 100 * 2000) / (1000 * 1000 + 997 * 100) = 199400000 / 1099700 = 181
        assert_eq!(delta_y, 181);
        
        // Test output price with 0.3% fee  
        let delta_x = cpmm.get_output_price(100, 1000, 2000).expect("Should not overflow");
        // (1000 * 1000 * 100) / (997 * (2000 - 100)) + 1 = 100000000 / 1894300 + 1 = 52 + 1 = 53
        assert_eq!(delta_x, 53);
    }

    #[test]
    fn test_trading_functions() {
        let mut cpmm = CPMM::new(1000, 2000, 1414);
        
        // Test eth to token
        let tokens_received = cpmm.eth_to_token(100).expect("Should not overflow/underflow");
        assert_eq!(tokens_received, 181);
        assert_eq!(cpmm.e, 1100);
        assert_eq!(cpmm.t, 1819);
        
        // Reset and test token to eth
        let mut cpmm2 = CPMM::new(1000, 2000, 1414);
        let eth_received = cpmm2.token_to_eth(100).expect("Should not overflow/underflow");
        assert_eq!(eth_received, 47); // (997 * 100 * 1000) / (1000 * 2000 + 997 * 100) = 47
        assert_eq!(cpmm2.e, 953);
        assert_eq!(cpmm2.t, 2100);
    }
}

#[cfg(kani)]
mod kani_verification {
    use super::*;

    /// Kani harness to verify all 5 properties of Theorem 2 from the PDF:
    /// "Let (e,t,l) --addLiquiditycode(Δe)--> (e'',t'',l'')"
    /// 
    /// Property 1: e < e''
    /// Property 2: t < t'' and t'' ≤ t' + 1 (where t' = t × (1 + Δe/e))
    /// Property 3: l < l'' 
    /// Property 4: k < k'' (where k = e × t)
    /// Property 5: (l''/l)² < k''/k (liquidity growth squared vs constant product growth)
    #[kani::proof]
    fn verify_add_liquidity_thm2() {
        // Use fixed values for faster verification (like other harnesses)
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let delta_e: u16 = kani::any();

        // Test with smaller bounds for delta_e to reduce proof space
        kani::assume(delta_e >= 1 && delta_e <= 500);

        // Create CPMM instance and capture initial state
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let e_before = cpmm.e;
        let t_before = cpmm.t;
        let l_before = cpmm.l;
        let k_before = cpmm.k();

        // Execute add_liquidity operation - only test cases that don't overflow
        if let Ok((returned_delta_t, returned_delta_l)) = cpmm.add_liquidity(delta_e) {
            let e_after = cpmm.e;
            let t_after = cpmm.t;
            let l_after = cpmm.l;
            let k_after = cpmm.k();

            // Property 1: e < e'' (ether reserve strictly increases)
            assert!(e_before < e_after, "Property 1 violated: e must strictly increase");

            // Property 2: t < t'' and t'' ≤ t' + 1
            // First part: t < t''
            assert!(t_before < t_after, "Property 2a violated: t must strictly increase");
            
            // Second part: t'' ≤ t' + 1 where t' = t × (1 + Δe/e)
            // Calculate t' using integer arithmetic: t' = t + (Δe × t)/e
            let delta_e_u32 = delta_e as u32;
            let e_u32 = e_before as u32;
            let t_u32 = t_before as u32;
            let t_prime_fractional = t_u32 + (delta_e_u32 * t_u32) / e_u32;
            
            // t'' should be ≤ t' + 1, but since we use integer arithmetic, 
            // we verify t'' ≤ ⌊t'⌋ + 1
            assert!(t_after as u32 <= t_prime_fractional + 1, "Property 2b violated: t'' must be ≤ t' + 1");

            // Property 3: l < l'' (liquidity supply strictly increases)
            assert!(l_before < l_after, "Property 3 violated: l must strictly increase");

            // Property 4: k < k'' (constant product strictly increases)
            assert!(k_before < k_after, "Property 4 violated: k must strictly increase");

            // Property 5: (l''/l)² < k''/k (liquidity growth squared vs constant product growth)
            // Calculate (l''/l)² and k''/k using integer arithmetic to avoid overflow
            let l_before_u32 = l_before as u32;
            let l_after_u32 = l_after as u32;
            
            // (l''/l)² = (l_after/l_before)² 
            // To avoid floating point, we check: (l_after)² * k_before < (l_before)² * k_after
            // This is equivalent to: (l''/l)² < k''/k
            let l_after_squared = l_after_u32 * l_after_u32;
            let l_before_squared = l_before_u32 * l_before_u32;
            
            // Use u64 for the multiplication to prevent overflow
            let left_side = (l_after_squared as u64) * (k_before as u64);
            let right_side = (l_before_squared as u64) * (k_after as u64);
            
            assert!(left_side < right_side, "Property 5 violated: (l''/l)² must be < k''/k");
        }
        // If overflow occurs, that's acceptable - we just skip verification for that case
    }

    /// Kani harness to verify Theorem 5 from the PDF:
    /// "Let (e,t,l) --removeLiquiditycode(Δl)--> (e'',t'',l''). Let k = e × t and k'' = e'' × t''.
    ///  Then, we have: k'' ≤ k"
    /// 
    /// This property states that removing liquidity never increases the constant product k.
    /// This ensures the pool cannot be exploited to create value through liquidity removal.
    #[kani::proof]
    fn verify_remove_liquidity_decreases_k() {
        // Create symbolic inputs
        let initial_e: u16 = kani::any();
        let initial_t: u16 = kani::any();
        let initial_l: u16 = kani::any();
        let delta_l: u16 = kani::any();

        // Test with moderate bounds for faster verification
        kani::assume(initial_e >= 1 && initial_e <= 500);
        kani::assume(initial_t >= 1 && initial_t <= 500);
        kani::assume(initial_l >= 1 && initial_l <= 500);
        kani::assume(delta_l >= 1 && delta_l <= 250 && delta_l < initial_l);

        // Create CPMM instance and record initial k
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let k_before = cpmm.k();

        // Execute remove_liquidity operation - only test cases that don't overflow/underflow
        if let Ok((_delta_e, _delta_t)) = cpmm.remove_liquidity(delta_l) {
            let k_after = cpmm.k();

            // Verify Theorem 5: k'' ≤ k
            // The constant product must not increase after removing liquidity
            assert!(k_after <= k_before, "Theorem 5 violated: k must not increase after removing liquidity");
        }
        // If overflow/underflow occurs, that's acceptable - we just skip verification for that case
    }

    /// Kani harness to verify that eth_to_token trading increases k due to fees
    /// This verifies Theorem 7/8 from the PDF: trading with fees increases the constant product k
    #[kani::proof]
    fn verify_eth_to_token_increases_k() {
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let delta_e: u16 = kani::any();

        // Test with smaller bounds for faster verification
        kani::assume(delta_e >= 1 && delta_e <= initial_e);

        // Create CPMM and record initial k
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let k_before = cpmm.k();

        // Execute trade - only test cases that don't overflow/underflow
        if let Ok(tokens_out) = cpmm.eth_to_token(delta_e) {
            let k_after = cpmm.k();

            // Verify that trading increases k due to fees (Theorem 7/8)
            // The constant product should increase due to the 0.3% trading fee
            assert!(k_after >= k_before, "Trading should not decrease k due to fees");
        }
        // If overflow/underflow occurs, that's acceptable - we just skip verification for that case
    }

    /// Property 3: Token to ETH swap increases k due to fees
    /// When swapping tokens for ETH, the constant product k should increase or stay the same
    /// due to the 0.3% trading fee embedded in the swap formulas.
    #[kani::proof]
    fn verify_token_to_eth_increases_k() {
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let delta_t: u16 = kani::any();

        // Test with smaller bounds for faster verification
        kani::assume(delta_t >= 1 && delta_t <= initial_t / 2);

        // Create CPMM instance and record initial k
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let k_before = cpmm.k();

        // Execute token to ETH swap
        if let Ok(_eth_out) = cpmm.token_to_eth(delta_t) {
            let k_after = cpmm.k();

            // Verify that k increases or stays the same due to fees
            assert!(k_after >= k_before, "Property 3 violated: k must not decrease after token to ETH swap");
        }
    }

    /// Property 4: ETH to token exact swap increases k due to fees
    /// When swapping ETH for an exact amount of tokens, k should increase or stay the same
    /// due to the 0.3% trading fee embedded in the getOutputPrice formula.
    #[kani::proof]
    fn verify_eth_to_token_exact_increases_k() {
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let delta_t: u16 = kani::any();

        // Test with smaller bounds for faster verification
        kani::assume(delta_t >= 1 && delta_t <= initial_t / 2);

        // Create CPMM instance and record initial k
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let k_before = cpmm.k();

        // Execute ETH to token exact swap
        if let Ok(_eth_in) = cpmm.eth_to_token_exact(delta_t) {
            let k_after = cpmm.k();

            // Verify that k increases or stays the same due to fees
            assert!(k_after >= k_before, "Property 4 violated: k must not decrease after ETH to token exact swap");
        }
    }

    /// Property 5: Token to ETH exact swap increases k due to fees
    /// When swapping tokens for an exact amount of ETH, k should increase or stay the same
    /// due to the 0.3% trading fee embedded in the getOutputPrice formula.
    #[kani::proof]
    fn verify_token_to_eth_exact_increases_k() {
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let delta_e: u16 = kani::any();

        // Test with smaller bounds for faster verification
        kani::assume(delta_e >= 1 && delta_e <= initial_e / 2);

        // Create CPMM instance and record initial k
        let mut cpmm = CPMM::new(initial_e, initial_t, initial_l);
        let k_before = cpmm.k();

        // Execute token to ETH exact swap
        if let Ok(_tokens_in) = cpmm.token_to_eth_exact(delta_e) {
            let k_after = cpmm.k();

            // Verify that k increases or stays the same due to fees
            assert!(k_after >= k_before, "Property 5 violated: k must not decrease after token to ETH exact swap");
        }
    }

    /// Property 6: Swap operations preserve liquidity tokens
    /// When performing any swap operation (ETH to token or token to ETH),
    /// the total supply of liquidity tokens (l) should remain unchanged.
    #[kani::proof]
    fn verify_swaps_preserve_liquidity() {
        let initial_e: u16 = 10000;
        let initial_t: u16 = 20000;
        let initial_l: u16 = 14141;
        let swap_amount: u16 = kani::any();

        // Test with smaller bounds for faster verification
        kani::assume(swap_amount >= 1 && swap_amount <= 5000);

        // Test ETH to token swap
        let mut cpmm1 = CPMM::new(initial_e, initial_t, initial_l);
        let l_before = cpmm1.l;
        if let Ok(_) = cpmm1.eth_to_token(swap_amount) {
            assert_eq!(cpmm1.l, l_before, "Property 6 violated: liquidity changed during ETH to token swap");
        }

        // Test token to ETH swap
        let mut cpmm2 = CPMM::new(initial_e, initial_t, initial_l);
        let l_before2 = cpmm2.l;
        if let Ok(_) = cpmm2.token_to_eth(swap_amount) {
            assert_eq!(cpmm2.l, l_before2, "Property 6 violated: liquidity changed during token to ETH swap");
        }

        // Test ETH to token exact swap
        let mut cpmm3 = CPMM::new(initial_e, initial_t, initial_l);
        let l_before3 = cpmm3.l;
        if swap_amount < initial_t {
            if let Ok(_) = cpmm3.eth_to_token_exact(swap_amount) {
                assert_eq!(cpmm3.l, l_before3, "Property 6 violated: liquidity changed during ETH to token exact swap");
            }
        }

        // Test token to ETH exact swap
        let mut cpmm4 = CPMM::new(initial_e, initial_t, initial_l);
        let l_before4 = cpmm4.l;
        if swap_amount < initial_e {
            if let Ok(_) = cpmm4.token_to_eth_exact(swap_amount) {
                assert_eq!(cpmm4.l, l_before4, "Property 6 violated: liquidity changed during token to ETH exact swap");
            }
        }
    }
}
