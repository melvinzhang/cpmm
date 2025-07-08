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

    def __init__(self, ether_reserve: int, token_reserve: int, initial_liquidity: int):
        """
        Initializes the CPMM with given reserves and liquidity.

        Args:
            ether_reserve (int): The initial amount of Ether in the pool.
            token_reserve (int): The initial amount of tokens in the pool.
            initial_liquidity (int): The initial amount of liquidity tokens.
        """
        self.e = np.uint32(ether_reserve)
        self.t = np.uint32(token_reserve)
        self.l = np.uint32(initial_liquidity)

        self.FEE_NUMERATOR = np.uint32(997)
        self.FEE_DENOMINATOR = np.uint32(1000)

    def __repr__(self) -> str:
        """Provides a string representation of the CPMM's state."""
        return f"CPMM(e={self.e}, t={self.t}, l={self.l}, k={self.k()})"

    def k(self) -> np.uint64:
        """Calculates the constant product 'k' using 64-bit precision."""
        return np.uint64(self.e) * np.uint64(self.t)

    # --- Section 2: Updating Liquidity ---

    def addLiquidity(self, delta_e: int) -> tuple[np.uint32, np.uint32]:
        """
        Mints liquidity. This follows Definition 2: addLiquidity_code.
        """
        delta_e = np.uint32(delta_e)
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

    def removeLiquidity(self, delta_l: int) -> tuple[np.uint32, np.uint32]:
        """
        Burns liquidity. This follows Definition 4: removeLiquidity_code.
        """
        delta_l = np.uint32(delta_l)
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

    def getInputPrice(self, delta_x: int, x_reserve: int, y_reserve: int) -> np.uint32:
        """
        Calculates output tokens for a given input. Follows Definition 6.
        Uses uint64 for intermediate steps with overflow protection.
        """
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

    def getOutputPrice(self, delta_y: int, x_reserve: int, y_reserve: int) -> np.uint32:
        """
        Calculates input tokens for a desired output. Follows Definition 8.
        Uses uint64 for intermediate steps with overflow protection.
        """
        if delta_y >= y_reserve:
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

    def ethToToken(self, delta_e: int) -> np.uint32:
        """Swaps Ether for tokens. Follows Section 4.1.2."""
        delta_t = self.getInputPrice(np.uint32(delta_e), self.e, self.t)
        self.e += np.uint32(delta_e)
        self.t -= delta_t
        return delta_t

    def tokenToEth(self, delta_t: int) -> np.uint32:
        """Swaps tokens for Ether. Follows Section 4.3.2."""
        delta_e = self.getInputPrice(np.uint32(delta_t), self.t, self.e)
        self.t += np.uint32(delta_t)
        self.e -= delta_e
        return delta_e
        
    def ethToTokenExact(self, delta_t: int) -> np.uint32:
        """Buys an exact amount of tokens with Ether. Follows Section 4.2.2."""
        delta_e = self.getOutputPrice(np.uint32(delta_t), self.e, self.t)
        self.e += delta_e
        self.t -= np.uint32(delta_t)
        return delta_e

    def tokenToEthExact(self, delta_e: int) -> np.uint32:
        """Buys an exact amount of Ether with tokens. Follows Section 4.4.2."""
        delta_t = self.getOutputPrice(np.uint32(delta_e), self.t, self.e)
        self.t += delta_t
        self.e -= np.uint32(delta_e)
        return delta_t


# --- Example Usage ---
if __name__ == '__main__':
    initial_eth = 20_000_000
    initial_token = 1_000_000_000
    initial_liquidity = 200_000_000
    
    pool = CPMM(initial_eth, initial_token, initial_liquidity)
    print("Initial Pool State:")
    print(pool)
    print("-" * 30)

    # --- Trading Operations ---
    print("1. Trading ETH for Tokens (ethToToken):")
    eth_in = 1_000_000
    tokens_out = pool.ethToToken(eth_in)
    print(f"Sold {eth_in} ETH and received {tokens_out} tokens.")
    print("Pool State after trade:")
    print(pool)
    print("-" * 30)
    
    # --- Overflow Example ---
    print("2. Testing Overflow Protection:")
    # Create a pool with large numbers close to the uint32 limit
    large_val = np.iinfo(np.uint32).max
    overflow_pool = CPMM(large_val, large_val, large_val)
    
    # This trade will cause the numerator in getInputPrice to exceed uint64
    trade_amount = large_val // 2
    
    try:
        print(f"Attempting a large trade of {trade_amount} that should overflow...")
        overflow_pool.ethToToken(trade_amount)
    except OverflowError as e:
        print(f"Successfully caught expected error: {e}")
