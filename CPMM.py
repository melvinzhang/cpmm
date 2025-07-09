import numpy as np

class CPMM:
    """
    A Python implementation of the Constant Product Market Maker (CPMM) model
    based on the formal specification by Yi Zhang, Xiaohong Chen, and Daejun Park.

    This class uses unsigned 32-bit integers for state variables (e, t, l) and
    unsigned 64-bit integers for intermediate calculations. It includes overflow
    checks to ensure mathematical correctness within the bounds of fixed-precision
    integers, adhering strictly to the integer-based formulas from the paper.

    Attributes:
        e (np.uint32): The amount of Ether reserve.
        t (np.uint32): The amount of Token reserve.
        l (np.uint32): The total amount of liquidity tokens.
        FEE_NUMERATOR (np.uint32): The numerator for the trading fee (e.g., 997).
        FEE_DENOMINATOR (np.uint32): The denominator for the trading fee (e.g., 1000).
    """

    @staticmethod
    def _verify_uint32(value, param_name: str) -> None:
        """
        Helper function to verify that a value is of type np.uint32.
        
        Args:
            value: The value to verify
            param_name (str): The parameter name for error messages
            
        Raises:
            TypeError: If value is not np.uint32
        """
        if not isinstance(value, np.uint32):
            raise TypeError(f"{param_name} must be np.uint32, got {type(value)}")

    def __init__(self, ether_reserve: np.uint32, token_reserve: np.uint32, initial_liquidity: np.uint32):
        """
        Initializes the CPMM with given reserves and liquidity.

        Args:
            ether_reserve (np.uint32): The initial amount of Ether in the pool.
            token_reserve (np.uint32): The initial amount of tokens in the pool.
            initial_liquidity (np.uint32): The initial amount of liquidity tokens.
        """
        # Type verification
        self._verify_uint32(ether_reserve, "ether_reserve")
        self._verify_uint32(token_reserve, "token_reserve")
        self._verify_uint32(initial_liquidity, "initial_liquidity")
            
        self.e = ether_reserve
        self.t = token_reserve
        self.l = initial_liquidity

        self.FEE_NUMERATOR = np.uint32(997)
        self.FEE_DENOMINATOR = np.uint32(1000)

    def __repr__(self) -> str:
        """Provides a string representation of the CPMM's state."""
        return f"CPMM(e={self.e}, t={self.t}, l={self.l}, k={self.k()})"

    def k(self) -> np.uint64:
        """Calculates the constant product 'k' using 64-bit precision."""
        return np.uint64(self.e) * np.uint64(self.t)

    # --- Section 2: Updating Liquidity ---

    def addLiquidity(self, delta_e: np.uint32) -> tuple[np.uint32, np.uint32]:
        """
        Mints liquidity. This follows Definition 2: addLiquidity_code.
        """
        # Type verification
        self._verify_uint32(delta_e, "delta_e")
            
        if self.e == 0 or self.t == 0:
            raise ValueError("Cannot add liquidity to an uninitialized pool (e=0 or t=0).")

        u64_delta_e = np.uint64(delta_e)
        u64_t = np.uint64(self.t)
        u64_e = np.uint64(self.e)
        u64_l = np.uint64(self.l)

        delta_t = np.uint32(((u64_delta_e * u64_t) // u64_e) + 1)
        delta_l = np.uint32((u64_delta_e * u64_l) // u64_e)

        self.e += delta_e
        self.t += delta_t
        self.l += delta_l

        return delta_t, delta_l

    def removeLiquidity(self, delta_l: np.uint32) -> tuple[np.uint32, np.uint32]:
        """
        Burns liquidity. This follows Definition 4: removeLiquidity_code.
        """
        # Type verification
        self._verify_uint32(delta_l, "delta_l")
            
        if delta_l > self.l:
            raise ValueError("Cannot remove more liquidity than exists.")

        u64_delta_l = np.uint64(delta_l)
        u64_e = np.uint64(self.e)
        u64_t = np.uint64(self.t)
        u64_l = np.uint64(self.l)

        delta_e = np.uint32((u64_delta_l * u64_e) // u64_l)
        delta_t = np.uint32((u64_delta_l * u64_t) // u64_l)

        self.e -= delta_e
        self.t -= delta_t
        self.l -= delta_l

        return delta_e, delta_t

    # --- Section 3: Token Price Calculation ---

    def getInputPrice(self, delta_x: np.uint32, x_reserve: np.uint32, y_reserve: np.uint32) -> np.uint32:
        """
        Calculates output tokens for a given input. Follows Definition 6.
        Uses post-fee calculation to avoid overflow issues.
        """
        # Type verification
        self._verify_uint32(delta_x, "delta_x")
        self._verify_uint32(x_reserve, "x_reserve")
        self._verify_uint32(y_reserve, "y_reserve")
        
        # Cast all inputs to uint64 for calculation
        u64_delta_x = np.uint64(delta_x)
        u64_x = np.uint64(x_reserve)
        u64_y = np.uint64(y_reserve)
        u64_fee_num = np.uint64(self.FEE_NUMERATOR)
        u64_fee_den = np.uint64(self.FEE_DENOMINATOR)

        # Step 1: Calculate effective input after fees (round DOWN)
        # Since fee_num < fee_den, delta_x_effective < delta_x
        delta_x_effective = (u64_delta_x * u64_fee_num) // u64_fee_den
        
        # Step 2: Calculate output (round DOWN)
        # No overflow possible: delta_x_effective < delta_x (uint32) and y is uint32
        numerator = delta_x_effective * u64_y
        denominator = u64_x + delta_x_effective
        
        if denominator == 0:
            return np.uint32(0)
            
        return np.uint32(numerator // denominator)

    def getOutputPrice(self, delta_y: np.uint32, x_reserve: np.uint32, y_reserve: np.uint32) -> np.uint32:
        """
        Calculates input tokens for a desired output. Follows Definition 8.
        Uses overflow-free calculation by applying fee adjustment to denominator.
        """
        # Type verification
        self._verify_uint32(delta_y, "delta_y")
        self._verify_uint32(x_reserve, "x_reserve")
        self._verify_uint32(y_reserve, "y_reserve")
            
        if np.uint64(delta_y) >= np.uint64(y_reserve):
            raise ValueError("Output amount must be less than the total reserve.")

        # Cast all inputs to uint64 for calculation
        u64_delta_y = np.uint64(delta_y)
        u64_x = np.uint64(x_reserve)
        u64_y = np.uint64(y_reserve)
        u64_fee_num = np.uint64(self.FEE_NUMERATOR)
        u64_fee_den = np.uint64(self.FEE_DENOMINATOR)

        # Calculate base exchange amount (no overflow: both uint32 range)
        base_numerator = u64_x * u64_delta_y
        base_denominator = u64_y - u64_delta_y
        
        # Apply fee adjustment to denominator (no overflow: uint32 * 997 fits in uint64)
        fee_adjusted_denominator = (base_denominator * u64_fee_num) // u64_fee_den
        
        if fee_adjusted_denominator == 0:
            raise ValueError("Cannot calculate output price, division by zero.")

        # Calculate required input with +1 for pool protection
        return np.uint32(base_numerator // fee_adjusted_denominator) + np.uint32(1)

    # --- Section 4: Trading Tokens ---

    def ethToToken(self, delta_e: np.uint32) -> np.uint32:
        """Swaps Ether for tokens. Follows Section 4.1.2."""
        # Type verification
        self._verify_uint32(delta_e, "delta_e")
            
        delta_t = self.getInputPrice(delta_e, self.e, self.t)
        self.e += delta_e
        self.t -= delta_t
        return delta_t

    def tokenToEth(self, delta_t: np.uint32) -> np.uint32:
        """Swaps tokens for Ether. Follows Section 4.3.2."""
        # Type verification
        self._verify_uint32(delta_t, "delta_t")
            
        delta_e = self.getInputPrice(delta_t, self.t, self.e)
        self.t += delta_t
        self.e -= delta_e
        return delta_e
        
    def ethToTokenExact(self, delta_t: np.uint32) -> np.uint32:
        """Buys an exact amount of tokens with Ether. Follows Section 4.2.2."""
        # Type verification
        self._verify_uint32(delta_t, "delta_t")
            
        delta_e = self.getOutputPrice(delta_t, self.e, self.t)
        self.e += delta_e
        self.t -= delta_t
        return delta_e

    def tokenToEthExact(self, delta_e: np.uint32) -> np.uint32:
        """Buys an exact amount of Ether with tokens. Follows Section 4.4.2."""
        # Type verification
        self._verify_uint32(delta_e, "delta_e")
            
        delta_t = self.getOutputPrice(delta_e, self.t, self.e)
        self.t += delta_t
        self.e -= delta_e
        return delta_t


# --- Example Usage ---
if __name__ == '__main__':
    initial_eth = np.uint32(20_000_000)
    initial_token = np.uint32(1_000_000_000)
    initial_liquidity = np.uint32(200_000_000)
    
    pool = CPMM(initial_eth, initial_token, initial_liquidity)
    print("Initial Pool State:")
    print(pool)
    print("-" * 30)

    # --- Trading Operations ---
    print("1. Trading ETH for Tokens (ethToToken):")
    eth_in = np.uint32(1_000_000)
    tokens_out = pool.ethToToken(eth_in)
    print(f"Sold {eth_in} ETH and received {tokens_out} tokens.")
    print("Pool State after trade:")
    print(pool)
    print("-" * 30)
    
    # --- Post-fee calculation eliminates overflow in getInputPrice ---
    print("2. Overflow Protection with Post-Fee Calculation:")
    # Create a pool with large numbers close to the uint32 limit
    large_val = np.iinfo(np.uint32).max
    overflow_pool = CPMM(np.uint32(large_val // 10), np.uint32(large_val // 10), np.uint32(large_val // 10))
    
    # This large trade would have caused overflow in the original implementation
    trade_amount = np.uint32(large_val // 100)
    
    print(f"Attempting a large trade of {trade_amount}...")
    tokens_out = overflow_pool.getInputPrice(trade_amount, overflow_pool.e, overflow_pool.t)
    print(f"Successfully calculated output: {tokens_out} tokens")
    print("The post-fee calculation prevents overflow in getInputPrice!")
