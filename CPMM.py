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

    def __init__(self, ether_reserve: np.uint32, token_reserve: np.uint32, initial_liquidity: np.uint32):
        """
        Initializes the CPMM with given reserves and liquidity.

        Args:
            ether_reserve (np.uint32): The initial amount of Ether in the pool.
            token_reserve (np.uint32): The initial amount of tokens in the pool.
            initial_liquidity (np.uint32): The initial amount of liquidity tokens.
        """
        # Type verification
        if not isinstance(ether_reserve, np.uint32):
            raise TypeError(f"ether_reserve must be np.uint32, got {type(ether_reserve)}")
        if not isinstance(token_reserve, np.uint32):
            raise TypeError(f"token_reserve must be np.uint32, got {type(token_reserve)}")
        if not isinstance(initial_liquidity, np.uint32):
            raise TypeError(f"initial_liquidity must be np.uint32, got {type(initial_liquidity)}")
            
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
        if not isinstance(delta_e, np.uint32):
            raise TypeError(f"delta_e must be np.uint32, got {type(delta_e)}")
            
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
        if not isinstance(delta_l, np.uint32):
            raise TypeError(f"delta_l must be np.uint32, got {type(delta_l)}")
            
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
        Uses uint64 for intermediate steps with overflow protection.
        """
        # Type verification
        if not isinstance(delta_x, np.uint32):
            raise TypeError(f"delta_x must be np.uint32, got {type(delta_x)}")
        if not isinstance(x_reserve, np.uint32):
            raise TypeError(f"x_reserve must be np.uint32, got {type(x_reserve)}")
        if not isinstance(y_reserve, np.uint32):
            raise TypeError(f"y_reserve must be np.uint32, got {type(y_reserve)}")
            
        u64_max = np.iinfo(np.uint64).max
        
        # Cast all inputs to uint64 for calculation
        u64_delta_x = np.uint64(delta_x)
        u64_x = np.uint64(x_reserve)
        u64_y = np.uint64(y_reserve)
        u64_fee_num = np.uint64(self.FEE_NUMERATOR)
        u64_fee_den = np.uint64(self.FEE_DENOMINATOR)

        # Numerator calculation: fee_num * delta_x * y
        # Check for overflow on the final multiplication
        num_part1 = u64_fee_num * u64_delta_x
        if u64_y > 0 and num_part1 > u64_max // u64_y:
            raise OverflowError("Numerator calculation exceeds uint64 capacity.")
        numerator = num_part1 * u64_y

        # Denominator calculation: (fee_den * x) + (fee_num * delta_x)
        den_part1 = u64_fee_den * u64_x
        den_part2 = u64_fee_num * u64_delta_x # Same as num_part1
        if den_part1 > u64_max - den_part2:
            raise OverflowError("Denominator calculation exceeds uint64 capacity.")
        denominator = den_part1 + den_part2
        
        if denominator == 0:
            return np.uint32(0)
            
        return np.uint32(numerator // denominator)

    def getOutputPrice(self, delta_y: np.uint32, x_reserve: np.uint32, y_reserve: np.uint32) -> np.uint32:
        """
        Calculates input tokens for a desired output. Follows Definition 8.
        Uses uint64 for intermediate steps with overflow protection.
        """
        # Type verification
        if not isinstance(delta_y, np.uint32):
            raise TypeError(f"delta_y must be np.uint32, got {type(delta_y)}")
        if not isinstance(x_reserve, np.uint32):
            raise TypeError(f"x_reserve must be np.uint32, got {type(x_reserve)}")
        if not isinstance(y_reserve, np.uint32):
            raise TypeError(f"y_reserve must be np.uint32, got {type(y_reserve)}")
            
        if np.uint64(delta_y) >= np.uint64(y_reserve):
            raise ValueError("Output amount must be less than the total reserve.")
        
        u64_max = np.iinfo(np.uint64).max

        # Cast all inputs to uint64 for calculation
        u64_delta_y = np.uint64(delta_y)
        u64_x = np.uint64(x_reserve)
        u64_y = np.uint64(y_reserve)
        u64_fee_num = np.uint64(self.FEE_NUMERATOR)
        u64_fee_den = np.uint64(self.FEE_DENOMINATOR)

        # Numerator calculation: fee_den * x * delta_y
        num_part1 = u64_fee_den * u64_x
        if u64_delta_y > 0 and num_part1 > u64_max // u64_delta_y:
            raise OverflowError("Numerator calculation exceeds uint64 capacity.")
        numerator = num_part1 * u64_delta_y

        # Denominator calculation: fee_num * (y - delta_y)
        denominator = u64_fee_num * (u64_y - u64_delta_y)
        
        if denominator == 0:
            raise ValueError("Cannot calculate output price, division by zero.")

        return np.uint32(numerator // denominator) + np.uint32(1)

    # --- Section 4: Trading Tokens ---

    def ethToToken(self, delta_e: np.uint32) -> np.uint32:
        """Swaps Ether for tokens. Follows Section 4.1.2."""
        # Type verification
        if not isinstance(delta_e, np.uint32):
            raise TypeError(f"delta_e must be np.uint32, got {type(delta_e)}")
            
        delta_t = self.getInputPrice(delta_e, self.e, self.t)
        self.e += delta_e
        self.t -= delta_t
        return delta_t

    def tokenToEth(self, delta_t: np.uint32) -> np.uint32:
        """Swaps tokens for Ether. Follows Section 4.3.2."""
        # Type verification
        if not isinstance(delta_t, np.uint32):
            raise TypeError(f"delta_t must be np.uint32, got {type(delta_t)}")
            
        delta_e = self.getInputPrice(delta_t, self.t, self.e)
        self.t += delta_t
        self.e -= delta_e
        return delta_e
        
    def ethToTokenExact(self, delta_t: np.uint32) -> np.uint32:
        """Buys an exact amount of tokens with Ether. Follows Section 4.2.2."""
        # Type verification
        if not isinstance(delta_t, np.uint32):
            raise TypeError(f"delta_t must be np.uint32, got {type(delta_t)}")
            
        delta_e = self.getOutputPrice(delta_t, self.e, self.t)
        self.e += delta_e
        self.t -= delta_t
        return delta_e

    def tokenToEthExact(self, delta_e: np.uint32) -> np.uint32:
        """Buys an exact amount of Ether with tokens. Follows Section 4.4.2."""
        # Type verification
        if not isinstance(delta_e, np.uint32):
            raise TypeError(f"delta_e must be np.uint32, got {type(delta_e)}")
            
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
    
    # --- Overflow Example ---
    print("2. Testing Overflow Protection:")
    # Create a pool with large numbers close to the uint32 limit
    large_val = np.iinfo(np.uint32).max
    overflow_pool = CPMM(np.uint32(large_val), np.uint32(large_val), np.uint32(large_val))
    
    # This trade will cause the numerator in getInputPrice to exceed uint64
    trade_amount = np.uint32(large_val // 2)
    
    try:
        print(f"Attempting a large trade of {trade_amount} that should overflow...")
        overflow_pool.ethToToken(trade_amount)
    except OverflowError as e:
        print(f"Successfully caught expected error: {e}")
