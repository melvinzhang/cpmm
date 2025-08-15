/// Constant Product Market Maker (CPMM) implementation
/// Based on the formal specification by Yi Zhang, Xiaohong Chen, and Daejun Park
/// 
/// This implementation follows the integer arithmetic versions (_code) from the PDF:
/// - Uses u32 for state variables (e, t, l)  
/// - Uses u64 for intermediate calculations
/// - Implements exact formulas from Definitions 2, 4, 6, and 8

#[derive(Debug, Clone)]
pub struct CPMM {
    /// Ether reserve (in wei)
    pub e: u32,
    /// Token reserve  
    pub t: u32,
    /// Total liquidity
    pub l: u32,
}

impl CPMM {
    /// Creates a new CPMM instance
    pub fn new(ether_reserve: u32, token_reserve: u32, liquidity: u32) -> Self {
        Self {
            e: ether_reserve,
            t: token_reserve,
            l: liquidity,
        }
    }

    /// Returns the constant product k = e × t
    pub fn k(&self) -> u64 {
        (self.e as u64) * (self.t as u64)
    }

    /// Definition 2: addLiquiditycode
    /// Takes integer Δe > 0 and updates state (e,t,l) ∈ Z³ → (e'',t'',l'') ∈ Z³
    /// 
    /// Where:
    /// e'' = e + Δe = (1 + α)e
    /// t'' = t + ⌊(Δe × t)/e⌋ + 1 = ⌊(1 + α)t⌋ + 1  
    /// l'' = l + ⌊(Δe × l)/e⌋ = ⌊(1 + α)l⌋
    pub fn add_liquidity(&mut self, delta_e: u32) -> (u32, u32) {
        if self.e == 0 {
            panic!("Cannot add liquidity when e = 0");
        }

        let delta_e_u64 = delta_e as u64;
        let e_u64 = self.e as u64;
        let t_u64 = self.t as u64;
        let l_u64 = self.l as u64;

        // Calculate deltas using integer division (floor)
        let delta_t = ((delta_e_u64 * t_u64) / e_u64) as u32 + 1;
        let delta_l = ((delta_e_u64 * l_u64) / e_u64) as u32;

        // Update state
        self.e += delta_e;
        self.t += delta_t;
        self.l += delta_l;

        (delta_t, delta_l)
    }

    /// Definition 4: removeLiquiditycode  
    /// Takes integer 0 < Δl < l and updates state (e,t,l) ∈ Z³ → (e'',t'',l'') ∈ Z³
    ///
    /// Where:
    /// e'' = e - ⌊(Δl × e)/l⌋ = ⌈(1 - α)e⌉
    /// t'' = t - ⌊(Δl × t)/l⌋ = ⌈(1 - α)t⌉  
    /// l'' = l - Δl = (1 - α)l
    pub fn remove_liquidity(&mut self, delta_l: u32) -> (u32, u32) {
        if delta_l >= self.l {
            panic!("Cannot remove more liquidity than exists");
        }

        let delta_l_u64 = delta_l as u64;
        let e_u64 = self.e as u64;
        let t_u64 = self.t as u64;
        let l_u64 = self.l as u64;

        // Calculate deltas using integer division (floor)
        let delta_e = ((delta_l_u64 * e_u64) / l_u64) as u32;
        let delta_t = ((delta_l_u64 * t_u64) / l_u64) as u32;

        // Update state
        self.e -= delta_e;
        self.t -= delta_t;
        self.l -= delta_l;

        (delta_e, delta_t)
    }

    /// Definition 6: getInputPricecode
    /// Let ρ be the trade fee. Takes input Δx > 0, x, y ∈ Z, outputs Δy ∈ Z
    /// 
    /// getInputPricecode(Δx)(x,y) = Δy = ⌊(αγ)/(1 + αγ) × y⌋
    /// where α = Δx/x and γ = 1 - ρ
    /// 
    /// Implementation: (997 * Δx * y) / (1000 * x + 997 * Δx)
    /// where / is integer division with truncation (floor)
    pub fn get_input_price(&self, delta_x: u32, x_reserve: u32, y_reserve: u32) -> u32 {
        if x_reserve == 0 {
            return 0;
        }

        let delta_x_u64 = delta_x as u64;
        let x_u64 = x_reserve as u64;
        let y_u64 = y_reserve as u64;

        // Using ρ = 0.003 (0.3% fee), so γ = 0.997
        // Formula: (997 * Δx * y) / (1000 * x + 997 * Δx)
        let numerator = 997u64 * delta_x_u64 * y_u64;
        let denominator = 1000u64 * x_u64 + 997u64 * delta_x_u64;

        if denominator == 0 {
            return 0;
        }

        (numerator / denominator) as u32
    }

    /// Definition 8: getOutputPricecode
    /// Let ρ be the trade fee. Takes input 0 < Δy < y, x, y ∈ Z, outputs Δx ∈ Z
    ///
    /// getOutputPricecode(Δy)(x,y) = Δx = ⌊(β/(1-β)) × (1/γ) × x⌋ + 1
    /// where β = Δy/y < 1 and γ = 1 - ρ
    ///
    /// Implementation: (1000 * x * Δy) / (997 * (y - Δy)) + 1  
    /// where / is integer division with truncation (floor)
    pub fn get_output_price(&self, delta_y: u32, x_reserve: u32, y_reserve: u32) -> u32 {
        if delta_y >= y_reserve {
            panic!("Output amount must be less than reserve");
        }

        let delta_y_u64 = delta_y as u64;
        let x_u64 = x_reserve as u64;
        let y_u64 = y_reserve as u64;

        // Using ρ = 0.003 (0.3% fee), so γ = 0.997
        // Formula: (1000 * x * Δy) / (997 * (y - Δy)) + 1
        let numerator = 1000u64 * x_u64 * delta_y_u64;
        let denominator = 997u64 * (y_u64 - delta_y_u64);

        if denominator == 0 {
            panic!("Division by zero in output price calculation");
        }

        ((numerator / denominator) as u32) + 1
    }

    /// Section 4.1.2: ethToTokencode
    /// Takes integer input Δe (Δe > 0) and updates state
    /// e'' = e + Δe
    /// t'' = t - getInputPricecode(Δe, e, t) = ⌊t'⌋
    pub fn eth_to_token(&mut self, delta_e: u32) -> u32 {
        let delta_t = self.get_input_price(delta_e, self.e, self.t);
        self.e += delta_e;
        self.t -= delta_t;
        delta_t
    }

    /// Section 4.2.2: ethToTokenExactcode  
    /// Takes integer input Δt (0 < Δt < t) and updates state
    /// t'' = t - Δt
    /// e'' = e + getOutputPricecode(Δt, e, t)
    pub fn eth_to_token_exact(&mut self, delta_t: u32) -> u32 {
        let delta_e = self.get_output_price(delta_t, self.e, self.t);
        self.e += delta_e;
        self.t -= delta_t;
        delta_e
    }

    /// Section 4.3.2: tokenToEthcode
    /// Takes integer input Δt (Δt > 0) and updates state  
    /// t'' = t + Δt
    /// e'' = e - getInputPricecode(Δt, t, e) = ⌊e'⌋
    pub fn token_to_eth(&mut self, delta_t: u32) -> u32 {
        let delta_e = self.get_input_price(delta_t, self.t, self.e);
        self.t += delta_t;
        self.e -= delta_e;
        delta_e
    }

    /// Section 4.4.2: tokenToEthExactcode
    /// Takes integer input Δe (0 < Δe < e) and updates state
    /// e'' = e - Δe  
    /// t'' = t + getOutputPricecode(Δe, t, e)
    pub fn token_to_eth_exact(&mut self, delta_e: u32) -> u32 {
        let delta_t = self.get_output_price(delta_e, self.t, self.e);
        self.t += delta_t;
        self.e -= delta_e;
        delta_t
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
        
        let (delta_t, delta_l) = cpmm.add_liquidity(100);
        
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
        
        let (delta_e, delta_t) = cpmm.remove_liquidity(100);
        
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
        let delta_y = cpmm.get_input_price(100, 1000, 2000);
        // (997 * 100 * 2000) / (1000 * 1000 + 997 * 100) = 199400000 / 1099700 = 181
        assert_eq!(delta_y, 181);
        
        // Test output price with 0.3% fee  
        let delta_x = cpmm.get_output_price(100, 1000, 2000);
        // (1000 * 1000 * 100) / (997 * (2000 - 100)) + 1 = 100000000 / 1894300 + 1 = 52 + 1 = 53
        assert_eq!(delta_x, 53);
    }

    #[test]
    fn test_trading_functions() {
        let mut cpmm = CPMM::new(1000, 2000, 1414);
        
        // Test eth to token
        let tokens_received = cpmm.eth_to_token(100);
        assert_eq!(tokens_received, 181);
        assert_eq!(cpmm.e, 1100);
        assert_eq!(cpmm.t, 1819);
        
        // Reset and test token to eth
        let mut cpmm2 = CPMM::new(1000, 2000, 1414);
        let eth_received = cpmm2.token_to_eth(100);
        assert_eq!(eth_received, 47); // (997 * 100 * 1000) / (1000 * 2000 + 997 * 100) = 47
        assert_eq!(cpmm2.e, 953);
        assert_eq!(cpmm2.t, 2100);
    }
}