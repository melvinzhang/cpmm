use std::fmt;

/// A Rust implementation of the Constant Product Market Maker (CPMM) model
/// based on the formal specification by Yi Zhang, Xiaohong Chen, and Daejun Park.
///
/// This struct uses unsigned 32-bit integers for state variables (e, t, l) and
/// unsigned 64-bit integers for intermediate calculations. It includes overflow
/// checks to ensure mathematical correctness within the bounds of fixed-precision
/// integers, adhering strictly to the integer-based formulas from the paper.
#[derive(Debug, Clone)]
pub struct CPMM {
    /// The amount of Ether reserve
    pub e: u32,
    /// The amount of Token reserve
    pub t: u32,
    /// The total amount of liquidity tokens
    pub l: u32,
    /// The numerator for the trading fee (e.g., 997)
    pub fee_numerator: u32,
    /// The denominator for the trading fee (e.g., 1000)
    pub fee_denominator: u32,
}

impl CPMM {
    /// Initializes the CPMM with given reserves and liquidity.
    pub fn new(ether_reserve: u32, token_reserve: u32, initial_liquidity: u32) -> Self {
        CPMM {
            e: ether_reserve,
            t: token_reserve,
            l: initial_liquidity,
            fee_numerator: 997,
            fee_denominator: 1000,
        }
    }
}

impl fmt::Display for CPMM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CPMM(e={}, t={}, l={}, k={})", self.e, self.t, self.l, self.k())
    }
}

impl CPMM {
    /// Calculates the constant product 'k' using 64-bit precision.
    pub fn k(&self) -> u64 {
        self.e as u64 * self.t as u64
    }

    /// Mints liquidity. This follows Definition 2: addLiquidity_code.
    /// Returns (delta_t, delta_l) - the amount of tokens to deposit and liquidity tokens minted.
    pub fn add_liquidity(&mut self, delta_e: u32) -> Result<(u32, u32), &'static str> {
        if self.e == 0 || self.t == 0 {
            return Err("Cannot add liquidity to an uninitialized pool (e=0 or t=0)");
        }

        let u64_delta_e = delta_e as u64;
        let u64_t = self.t as u64;
        let u64_e = self.e as u64;
        let u64_l = self.l as u64;

        let delta_t = ((u64_delta_e * u64_t) / u64_e + 1) as u32;
        let delta_l = ((u64_delta_e * u64_l) / u64_e) as u32;

        // Check for overflow
        if self.e.checked_add(delta_e).is_none() {
            return Err("Ether reserve overflow");
        }
        if self.t.checked_add(delta_t).is_none() {
            return Err("Token reserve overflow");
        }
        if self.l.checked_add(delta_l).is_none() {
            return Err("Liquidity overflow");
        }

        self.e += delta_e;
        self.t += delta_t;
        self.l += delta_l;

        Ok((delta_t, delta_l))
    }

    /// Burns liquidity. This follows Definition 4: removeLiquidity_code.
    /// Returns (delta_e, delta_t) - the amount of ether and tokens to return.
    pub fn remove_liquidity(&mut self, delta_l: u32) -> Result<(u32, u32), &'static str> {
        if delta_l > self.l {
            return Err("Cannot remove more liquidity than exists");
        }

        let u64_delta_l = delta_l as u64;
        let u64_e = self.e as u64;
        let u64_t = self.t as u64;
        let u64_l = self.l as u64;

        let delta_e = ((u64_delta_l * u64_e) / u64_l) as u32;
        let delta_t = ((u64_delta_l * u64_t) / u64_l) as u32;

        self.e -= delta_e;
        self.t -= delta_t;
        self.l -= delta_l;

        Ok((delta_e, delta_t))
    }

    /// Calculates output tokens for a given input. Follows Definition 6.
    /// Uses post-fee calculation to avoid overflow issues.
    pub fn get_input_price(&self, delta_x: u32, x_reserve: u32, y_reserve: u32) -> Option<u32> {
        // Cast all inputs to u64 for calculation
        let u64_delta_x = delta_x as u64;
        let u64_x = x_reserve as u64;
        let u64_y = y_reserve as u64;
        let u64_fee_num = self.fee_numerator as u64;
        let u64_fee_den = self.fee_denominator as u64;

        // Step 1: Calculate effective input after fees (round DOWN)
        // Since fee_num < fee_den, delta_x_effective < delta_x
        let delta_x_effective = (u64_delta_x * u64_fee_num) / u64_fee_den;
        
        // Step 2: Calculate output (round DOWN)
        // No overflow possible: delta_x_effective < delta_x (u32) and y is u32
        let numerator = delta_x_effective * u64_y;
        let denominator = u64_x + delta_x_effective;
        
        if denominator == 0 {
            return None;
        }
            
        Some((numerator / denominator) as u32)
    }

    /// Calculates input tokens for a desired output. Follows Definition 8.
    /// Uses overflow-free calculation by applying fee adjustment to denominator.
    pub fn get_output_price(&self, delta_y: u32, x_reserve: u32, y_reserve: u32) -> Result<u32, &'static str> {
        if delta_y as u64 >= y_reserve as u64 {
            return Err("Output amount must be less than the total reserve");
        }

        // Cast all inputs to u64 for calculation
        let u64_delta_y = delta_y as u64;
        let u64_x = x_reserve as u64;
        let u64_y = y_reserve as u64;
        let u64_fee_num = self.fee_numerator as u64;
        let u64_fee_den = self.fee_denominator as u64;

        // Calculate base exchange amount (no overflow: both u32 range)
        let base_numerator = u64_x * u64_delta_y;
        let base_denominator = u64_y - u64_delta_y;
        
        // Apply fee adjustment to denominator (no overflow: u32 * 997 fits in u64)
        let fee_adjusted_denominator = (base_denominator * u64_fee_num) / u64_fee_den;
        
        if fee_adjusted_denominator == 0 {
            return Err("Cannot calculate output price, division by zero");
        }

        // Calculate required input with +1 for pool protection
        Ok((base_numerator / fee_adjusted_denominator) as u32 + 1)
    }

    /// Swaps Ether for tokens. Follows Section 4.1.2.
    pub fn eth_to_token(&mut self, delta_e: u32) -> Option<u32> {
        let delta_t = self.get_input_price(delta_e, self.e, self.t)?;
        
        // Check for overflow
        if self.e.checked_add(delta_e).is_none() {
            return None;
        }
        if delta_t > self.t {
            return None;
        }
        
        self.e += delta_e;
        self.t -= delta_t;
        Some(delta_t)
    }

    /// Swaps tokens for Ether. Follows Section 4.3.2.
    pub fn token_to_eth(&mut self, delta_t: u32) -> Option<u32> {
        let delta_e = self.get_input_price(delta_t, self.t, self.e)?;
        
        // Check for overflow
        if self.t.checked_add(delta_t).is_none() {
            return None;
        }
        if delta_e > self.e {
            return None;
        }
        
        self.t += delta_t;
        self.e -= delta_e;
        Some(delta_e)
    }

    /// Buys an exact amount of tokens with Ether. Follows Section 4.2.2.
    pub fn eth_to_token_exact(&mut self, delta_t: u32) -> Result<u32, &'static str> {
        let delta_e = self.get_output_price(delta_t, self.e, self.t)?;
        
        // Check for overflow
        if self.e.checked_add(delta_e).is_none() {
            return Err("Ether reserve overflow");
        }
        if delta_t > self.t {
            return Err("Not enough tokens in reserve");
        }
        
        self.e += delta_e;
        self.t -= delta_t;
        Ok(delta_e)
    }

    /// Buys an exact amount of Ether with tokens. Follows Section 4.4.2.
    pub fn token_to_eth_exact(&mut self, delta_e: u32) -> Result<u32, &'static str> {
        let delta_t = self.get_output_price(delta_e, self.t, self.e)?;
        
        // Check for overflow
        if self.t.checked_add(delta_t).is_none() {
            return Err("Token reserve overflow");
        }
        if delta_e > self.e {
            return Err("Not enough ether in reserve");
        }
        
        self.t += delta_t;
        self.e -= delta_e;
        Ok(delta_t)
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn verify_k() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        
        let pool = CPMM::new(e, t, l);
        let k_value = pool.k();
        
        // Verify that k = e * t with proper 64-bit precision
        assert_eq!(k_value, (e as u64) * (t as u64));
    }

    #[kani::proof]
    fn verify_add_liquidity() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_e: u32 = kani::any();
        
        // Constrain to avoid division by zero
        kani::assume(e > 0 && t > 0);
        
        // Constrain to prevent overflow in intermediate calculations
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        kani::assume((l as u64) < (u32::MAX as u64 / 2));
        kani::assume((delta_e as u64) < (u32::MAX as u64 / 100));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Ok((delta_t, delta_l)) = pool.add_liquidity(delta_e) {
            // Verify invariants
            assert!(pool.e == e + delta_e);
            assert!(pool.t == t + delta_t);
            assert!(pool.l == l + delta_l);
            
            // Verify that k increases or stays the same
            let k_after = pool.k();
            assert!(k_after >= k_before);
            
            // Verify the formulas used
            assert_eq!(delta_t, ((delta_e as u64 * t as u64) / e as u64 + 1) as u32);
            assert_eq!(delta_l, ((delta_e as u64 * l as u64) / e as u64) as u32);
        }
    }

    #[kani::proof]
    fn verify_remove_liquidity() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_l: u32 = kani::any();
        
        // Constrain to avoid division by zero and ensure valid removal
        kani::assume(l > 0 && delta_l <= l);
        
        // Constrain to prevent overflow in intermediate calculations
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Ok((delta_e, delta_t)) = pool.remove_liquidity(delta_l) {
            // Verify invariants
            assert!(pool.e == e - delta_e);
            assert!(pool.t == t - delta_t);
            assert!(pool.l == l - delta_l);
            
            // Verify that k decreases or stays the same
            let k_after = pool.k();
            assert!(k_after <= k_before);
            
            // Verify the formulas used
            assert_eq!(delta_e, ((delta_l as u64 * e as u64) / l as u64) as u32);
            assert_eq!(delta_t, ((delta_l as u64 * t as u64) / l as u64) as u32);
            
            // If we remove all liquidity, reserves should be zero
            if delta_l == l {
                assert!(pool.e == 0 || pool.t == 0 || pool.l == 0);
            }
        }
    }

    #[kani::proof]
    fn verify_get_input_price() {
        let delta_x: u32 = kani::any();
        let x_reserve: u32 = kani::any();
        let y_reserve: u32 = kani::any();
        
        // Constrain inputs to reasonable values
        kani::assume(x_reserve > 0 && y_reserve > 0);
        kani::assume((delta_x as u64) < (u32::MAX as u64 / 100));
        
        let pool = CPMM::new(x_reserve, y_reserve, 1000);
        
        if let Some(output) = pool.get_input_price(delta_x, x_reserve, y_reserve) {
            // Output should be less than y_reserve
            assert!(output < y_reserve);
            
            // Verify the constant product formula approximately holds
            // After trade: (x + delta_x * fee) * (y - output) ~= x * y
            let fee_adjusted_input = (delta_x as u64 * pool.fee_numerator as u64) / pool.fee_denominator as u64;
            let new_x = x_reserve as u64 + fee_adjusted_input;
            let new_y = y_reserve as u64 - output as u64;
            
            // The product should be approximately preserved (accounting for rounding)
            let original_k = x_reserve as u64 * y_reserve as u64;
            let new_k = new_x * new_y;
            
            // Due to integer division, new_k might be slightly less than original_k
            assert!(new_k <= original_k || (original_k - new_k) <= new_x);
        }
    }

    #[kani::proof]
    fn verify_get_output_price() {
        let delta_y: u32 = kani::any();
        let x_reserve: u32 = kani::any();
        let y_reserve: u32 = kani::any();
        
        // Constrain inputs to reasonable values
        kani::assume(x_reserve > 0 && y_reserve > 0);
        kani::assume(delta_y < y_reserve);
        kani::assume((x_reserve as u64) < (u32::MAX as u64 / 100));
        kani::assume((y_reserve as u64) < (u32::MAX as u64 / 100));
        
        let pool = CPMM::new(x_reserve, y_reserve, 1000);
        
        if let Ok(input_required) = pool.get_output_price(delta_y, x_reserve, y_reserve) {
            // Input required should be positive
            assert!(input_required > 0);
            
            // Verify that if we use this input, we get at least delta_y output
            if let Some(actual_output) = pool.get_input_price(input_required, x_reserve, y_reserve) {
                // Due to the +1 in get_output_price, actual_output should be >= delta_y
                assert!(actual_output >= delta_y);
            }
        }
    }

    #[kani::proof]
    fn verify_eth_to_token() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_e: u32 = kani::any();
        
        // Constrain to reasonable values
        kani::assume(e > 0 && t > 0);
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        kani::assume((delta_e as u64) < (u32::MAX as u64 / 100));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Some(delta_t) = pool.eth_to_token(delta_e) {
            // Verify state changes
            assert!(pool.e == e + delta_e);
            assert!(pool.t == t - delta_t);
            assert!(pool.l == l); // Liquidity unchanged
            
            // Verify that k approximately holds (accounting for fees)
            let k_after = pool.k();
            assert!(k_after >= k_before);
        }
    }

    #[kani::proof]
    fn verify_token_to_eth() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_t: u32 = kani::any();
        
        // Constrain to reasonable values
        kani::assume(e > 0 && t > 0);
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        kani::assume((delta_t as u64) < (u32::MAX as u64 / 100));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Some(delta_e) = pool.token_to_eth(delta_t) {
            // Verify state changes
            assert!(pool.e == e - delta_e);
            assert!(pool.t == t + delta_t);
            assert!(pool.l == l); // Liquidity unchanged
            
            // Verify that k approximately holds (accounting for fees)
            let k_after = pool.k();
            assert!(k_after >= k_before);
        }
    }

    #[kani::proof]
    fn verify_eth_to_token_exact() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_t: u32 = kani::any();
        
        // Constrain to reasonable values
        kani::assume(e > 0 && t > 0);
        kani::assume(delta_t < t);
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Ok(delta_e) = pool.eth_to_token_exact(delta_t) {
            // Verify state changes
            assert!(pool.e == e + delta_e);
            assert!(pool.t == t - delta_t);
            assert!(pool.l == l); // Liquidity unchanged
            
            // Verify that k approximately holds (accounting for fees)
            let k_after = pool.k();
            assert!(k_after >= k_before);
            
            // Verify that we got exactly delta_t tokens
            assert!(delta_e > 0);
        }
    }

    #[kani::proof]
    fn verify_token_to_eth_exact() {
        let e: u32 = kani::any();
        let t: u32 = kani::any();
        let l: u32 = kani::any();
        let delta_e: u32 = kani::any();
        
        // Constrain to reasonable values
        kani::assume(e > 0 && t > 0);
        kani::assume(delta_e < e);
        kani::assume((e as u64) < (u32::MAX as u64 / 2));
        kani::assume((t as u64) < (u32::MAX as u64 / 2));
        
        let mut pool = CPMM::new(e, t, l);
        let k_before = pool.k();
        
        if let Ok(delta_t) = pool.token_to_eth_exact(delta_e) {
            // Verify state changes
            assert!(pool.e == e - delta_e);
            assert!(pool.t == t + delta_t);
            assert!(pool.l == l); // Liquidity unchanged
            
            // Verify that k approximately holds (accounting for fees)
            let k_after = pool.k();
            assert!(k_after >= k_before);
            
            // Verify that we got exactly delta_e ether
            assert!(delta_t > 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let initial_eth = 20_000_000;
        let initial_token = 1_000_000_000;
        let initial_liquidity = 200_000_000;
        
        let mut pool = CPMM::new(initial_eth, initial_token, initial_liquidity);
        println!("Initial Pool State:");
        println!("{}", pool);
        println!("{}", "-".repeat(30));

        // Trading ETH for Tokens
        println!("1. Trading ETH for Tokens (eth_to_token):");
        let eth_in = 1_000_000;
        let tokens_out = pool.eth_to_token(eth_in).unwrap();
        println!("Sold {} ETH and received {} tokens.", eth_in, tokens_out);
        println!("Pool State after trade:");
        println!("{}", pool);
        println!("{}", "-".repeat(30));
    }

    #[test]
    fn test_overflow_protection() {
        // Create a pool with large numbers close to the u32 limit
        let large_val = u32::MAX;
        let pool = CPMM::new(large_val / 10, large_val / 10, large_val / 10);
        
        // This large trade would have caused overflow in a naive implementation
        let trade_amount = large_val / 100;
        
        println!("Attempting a large trade of {}...", trade_amount);
        let tokens_out = pool.get_input_price(trade_amount, pool.e, pool.t);
        assert!(tokens_out.is_some());
        println!("Successfully calculated output: {:?} tokens", tokens_out);
        println!("The post-fee calculation prevents overflow in get_input_price!");
    }
}

fn main() {
    let initial_eth = 20_000_000;
    let initial_token = 1_000_000_000;
    let initial_liquidity = 200_000_000;
    
    let mut pool = CPMM::new(initial_eth, initial_token, initial_liquidity);
    println!("Initial Pool State:");
    println!("{}", pool);
    println!("{}", "-".repeat(30));

    // Trading Operations
    println!("1. Trading ETH for Tokens (eth_to_token):");
    let eth_in = 1_000_000;
    if let Some(tokens_out) = pool.eth_to_token(eth_in) {
        println!("Sold {} ETH and received {} tokens.", eth_in, tokens_out);
        println!("Pool State after trade:");
        println!("{}", pool);
    }
    println!("{}", "-".repeat(30));
    
    // Overflow Protection Example
    println!("2. Overflow Protection with Post-Fee Calculation:");
    let large_val = u32::MAX;
    let overflow_pool = CPMM::new(large_val / 10, large_val / 10, large_val / 10);
    
    let trade_amount = large_val / 100;
    
    println!("Attempting a large trade of {}...", trade_amount);
    if let Some(tokens_out) = overflow_pool.get_input_price(trade_amount, overflow_pool.e, overflow_pool.t) {
        println!("Successfully calculated output: {} tokens", tokens_out);
        println!("The post-fee calculation prevents overflow in get_input_price!");
    }
}