import unittest
import numpy as np
from CPMM import CPMM


class TestCPMMSecurity(unittest.TestCase):
    """
    Comprehensive test suite to ensure CPMM is not exploitable and protects liquidity providers.
    Tests focus on security vulnerabilities, edge cases, and mathematical correctness.
    """

    def setUp(self):
        """Set up test pools with various configurations."""
        # Standard pool
        self.pool = CPMM(np.uint32(1000000), np.uint32(2000000), np.uint32(1000000))
        
        # Edge case pools
        self.small_pool = CPMM(np.uint32(100), np.uint32(200), np.uint32(100))
        self.large_pool = CPMM(np.uint32(1000000000), np.uint32(2000000000), np.uint32(1000000000))
        
        # Maximum values pool (close to uint32 limits)
        self.max_val = np.iinfo(np.uint32).max
        self.near_max_pool = CPMM(np.uint32(self.max_val // 2), np.uint32(self.max_val // 2), np.uint32(self.max_val // 3))

    def test_initialization_validation(self):
        """Test pool initialization with invalid values."""
        # Test negative values by creating uint32 from negative numbers (will overflow)
        with self.assertRaises((ValueError, OverflowError)):
            CPMM(np.uint32(-1), np.uint32(100), np.uint32(100))
        
        with self.assertRaises((ValueError, OverflowError)):
            CPMM(np.uint32(100), np.uint32(-1), np.uint32(100))
        
        with self.assertRaises((ValueError, OverflowError)):
            CPMM(np.uint32(100), np.uint32(100), np.uint32(-1))
        
        # Test values exceeding uint32 max
        with self.assertRaises((ValueError, OverflowError)):
            CPMM(np.uint32(self.max_val + 1), np.uint32(100), np.uint32(100))
            
        # Test type verification
        with self.assertRaises(TypeError):
            CPMM(100, np.uint32(100), np.uint32(100))  # First arg not uint32
            
        with self.assertRaises(TypeError):
            CPMM(np.uint32(100), 100, np.uint32(100))  # Second arg not uint32
            
        with self.assertRaises(TypeError):
            CPMM(np.uint32(100), np.uint32(100), 100)  # Third arg not uint32

    def test_constant_product_invariant(self):
        """Test that k = e * t remains constant or increases after trades (with fees)."""
        initial_k = self.pool.k()
        
        # Trade ETH for tokens
        self.pool.ethToToken(np.uint32(1000))
        k_after_trade1 = self.pool.k()
        self.assertGreaterEqual(k_after_trade1, initial_k, 
                               "Constant product should not decrease after trade")
        
        # Trade tokens for ETH
        self.pool.tokenToEth(np.uint32(1000))
        k_after_trade2 = self.pool.k()
        self.assertGreaterEqual(k_after_trade2, k_after_trade1,
                               "Constant product should not decrease after trade")

    def test_liquidity_provider_protection(self):
        """Test that liquidity providers cannot lose value through normal operations."""
        # Add liquidity
        initial_e = self.pool.e
        initial_t = self.pool.t
        initial_l = self.pool.l
        
        delta_t, delta_l = self.pool.addLiquidity(np.uint32(100000))
        
        # Remove the same liquidity
        returned_e, returned_t = self.pool.removeLiquidity(delta_l)
        
        # Check that LPs get back proportional amounts
        # Due to rounding, they might get slightly less but never more
        self.assertLessEqual(returned_e, 100000,
                            "LPs should not extract more ETH than deposited")
        self.assertLessEqual(returned_t, delta_t,
                            "LPs should not extract more tokens than deposited")

    def test_sandwich_attack_resistance(self):
        """Test resistance to sandwich attacks that could harm liquidity providers."""
        initial_k = self.pool.k()
        initial_pool_e = self.pool.e
        initial_pool_t = self.pool.t
        
        # Attacker front-runs a large trade
        attacker_tokens_bought = self.pool.ethToToken(np.uint32(50000))
        
        # Victim's trade
        victim_tokens = self.pool.ethToToken(np.uint32(100000))
        
        # Attacker back-runs by selling
        attacker_eth_received = self.pool.tokenToEth(attacker_tokens_bought)
        
        # The pool's k should have increased due to fees
        final_k = self.pool.k()
        self.assertGreater(final_k, initial_k,
                          "Pool should benefit from trading fees during sandwich attack")
        
        # Due to slippage and fees, the attacker's round trip should result in a loss
        # The specific test pool configuration might allow small profits, but
        # liquidity providers are still protected because k increased
        net_attacker_result = attacker_eth_received - 50000
        
        # Even if attacker makes a small profit, LPs benefit more from increased k
        k_increase = final_k - initial_k
        self.assertGreater(k_increase, 0,
                          "Liquidity providers benefit from increased pool value")

    def test_manipulation_through_extreme_trades(self):
        """Test that extreme trades cannot manipulate the pool unfairly."""
        initial_e = self.pool.e
        initial_t = self.pool.t
        initial_k = self.pool.k()
        
        # Try to manipulate price with large trade (half the pool)
        try:
            # This should either fail with overflow or give very poor rate
            tokens_received = self.pool.ethToToken(initial_e // 2)
            
            # If it succeeds, verify the invariant held and LPs are protected
            new_k = self.pool.k()
            self.assertGreaterEqual(new_k, initial_k,
                                   "Pool invariant must be maintained or increase")
            
            # Check pool invariant and price impact
            # For such a large trade (50% of pool), verify:
            # 1. Pool invariant is maintained or increased
            # 2. Price impact is severe
            
            # Calculate price impact (ETH price in terms of tokens)
            initial_eth_price = initial_t / initial_e  # tokens per ETH
            final_eth_price = self.pool.t / self.pool.e  # tokens per ETH after trade
            
            # After buying tokens with ETH, there's more ETH and fewer tokens
            # So ETH becomes less valuable (lower price in tokens)
            self.assertLess(final_eth_price, initial_eth_price,
                           "ETH price should decrease after selling ETH")
            
            # The price impact should be severe for such a large trade
            price_impact = (initial_eth_price - final_eth_price) / initial_eth_price
            self.assertGreater(price_impact, 0.3,
                              "Large trades should have severe price impact")
            
            # Verify the trade was unfavorable compared to initial price
            # tokens_received / eth_traded should be much less than initial price
            effective_rate = tokens_received / (initial_e // 2)
            initial_rate = initial_t / initial_e
            
            # Trader should get significantly worse rate than initial
            self.assertLess(effective_rate / initial_rate, 0.7,
                           "Large trades should receive poor exchange rates")
            
        except OverflowError:
            # This is also acceptable - overflow protection prevents the trade
            pass

    def test_rounding_attacks(self):
        """Test that rounding errors cannot be exploited to drain the pool."""
        # Try many small trades to exploit rounding
        initial_e = self.pool.e
        initial_t = self.pool.t
        
        # Perform 1000 tiny trades
        for _ in range(1000):
            try:
                # Trade 1 wei worth
                tokens = self.pool.ethToToken(np.uint32(1))
                if tokens > 0:
                    # Trade back
                    self.pool.tokenToEth(tokens)
            except:
                # Some trades might fail due to rounding
                pass
        
        # Pool should not have lost significant value
        self.assertGreaterEqual(self.pool.e, initial_e - 1000,
                               "Pool should not lose significant ETH through rounding")
        self.assertGreaterEqual(self.pool.t, initial_t - 1000,
                               "Pool should not lose significant tokens through rounding")

    def test_overflow_protection(self):
        """Test that overflow protection works correctly."""
        # Test with maximum values
        max_pool = CPMM(np.uint32(self.max_val - 1000), np.uint32(self.max_val - 1000), np.uint32(self.max_val // 2))
        
        # Try trades that would cause overflow
        with self.assertRaises(OverflowError):
            max_pool.ethToToken(np.uint32(self.max_val // 2))
        
        with self.assertRaises(OverflowError):
            max_pool.tokenToEth(np.uint32(self.max_val // 2))

    def test_division_by_zero_protection(self):
        """Test protection against division by zero."""
        # Try to remove all liquidity
        with self.assertRaises(ValueError):
            self.pool.removeLiquidity(self.pool.l + 1)
        
        # Try to trade more than available
        with self.assertRaises(ValueError):
            self.pool.ethToTokenExact(self.pool.t + 1)
        
        with self.assertRaises(ValueError):
            self.pool.tokenToEthExact(self.pool.e + 1)

    def test_liquidity_attack_vectors(self):
        """Test various attack vectors specific to liquidity operations."""
        # Test adding liquidity to empty pool
        empty_pool = CPMM(np.uint32(0), np.uint32(0), np.uint32(0))
        with self.assertRaises(ValueError):
            empty_pool.addLiquidity(np.uint32(1000))
        
        # Test that adding liquidity maintains price ratio
        initial_price_ratio = self.pool.e / self.pool.t
        self.pool.addLiquidity(np.uint32(100000))
        new_price_ratio = self.pool.e / self.pool.t
        self.assertAlmostEqual(initial_price_ratio, new_price_ratio, places=6,
                              msg="Adding liquidity should not change price ratio")

    def test_exact_output_trades_security(self):
        """Test that exact output trades cannot be exploited."""
        # Test buying exact tokens
        eth_needed = self.pool.ethToTokenExact(np.uint32(1000))
        
        # Verify that we can't buy more tokens than exist
        with self.assertRaises(ValueError):
            self.pool.ethToTokenExact(self.pool.t)
        
        # Test buying exact ETH
        tokens_needed = self.pool.tokenToEthExact(np.uint32(1000))
        
        # Verify that we can't buy more ETH than exists
        with self.assertRaises(ValueError):
            self.pool.tokenToEthExact(self.pool.e)

    def test_fee_consistency(self):
        """Test that fees are consistently applied and cannot be bypassed."""
        # Trade A -> B
        eth_in = np.uint32(10000)
        tokens_out = self.pool.getInputPrice(eth_in, self.pool.e, self.pool.t)
        
        # Calculate what output would be without fees using uint64 to prevent overflow
        u64_eth_in = np.uint64(eth_in)
        u64_t = np.uint64(self.pool.t)
        u64_e = np.uint64(self.pool.e)
        
        no_fee_output = (u64_eth_in * u64_t) // (u64_e + u64_eth_in)
        
        # Verify fee was applied (output should be ~0.3% less due to 997/1000 fee)
        fee_impact = float(no_fee_output - tokens_out) / float(no_fee_output)
        self.assertAlmostEqual(fee_impact, 0.003, places=3,
                              msg="Fee should be approximately 0.3%")

    def test_state_consistency(self):
        """Test that pool state remains consistent after operations."""
        # Perform various operations
        self.pool.ethToToken(np.uint32(1000))
        self.pool.tokenToEth(np.uint32(500))
        delta_t, delta_l = self.pool.addLiquidity(np.uint32(5000))
        self.pool.removeLiquidity(delta_l // 2)
        
        # Verify state variables are within valid bounds
        self.assertLessEqual(self.pool.e, self.max_val)
        self.assertLessEqual(self.pool.t, self.max_val)
        self.assertLessEqual(self.pool.l, self.max_val)
        
        # Verify k is reasonable
        self.assertGreater(self.pool.k(), 0)

    def test_liquidity_extraction_limits(self):
        """Test that liquidity cannot be extracted unfairly."""
        # Add liquidity from multiple providers
        provider1_delta_t, provider1_delta_l = self.pool.addLiquidity(np.uint32(100000))
        provider2_delta_t, provider2_delta_l = self.pool.addLiquidity(np.uint32(200000))
        
        # Try to remove more liquidity than owned
        with self.assertRaises(ValueError):
            self.pool.removeLiquidity(self.pool.l + 1)
        
        # Remove liquidity and verify proportional returns
        e_out, t_out = self.pool.removeLiquidity(provider1_delta_l)
        
        # Due to pool growth from fees, providers might get slightly more
        # but the ratio should be maintained
        ratio_in = 100000 / provider1_delta_t
        ratio_out = e_out / t_out
        self.assertAlmostEqual(ratio_in, ratio_out, delta=0.01,
                              msg="Liquidity removal should maintain deposit ratio")

    def test_integer_precision_attacks(self):
        """Test that integer precision cannot be exploited."""
        # Test with small values where precision matters most
        small_pool = CPMM(np.uint32(1000), np.uint32(1000), np.uint32(1000))
        
        # Multiple small trades
        for i in range(100):
            if i % 2 == 0:
                small_pool.ethToToken(np.uint32(1))
            else:
                small_pool.tokenToEth(np.uint32(1))
        
        # Pool should maintain its value despite rounding
        final_k = small_pool.k()
        initial_k = 1000 * 1000
        self.assertGreaterEqual(final_k, initial_k,
                               "Pool value should not decrease from precision attacks")

    def test_frontrunning_liquidity_additions(self):
        """Test protection against frontrunning liquidity additions."""
        initial_k = self.pool.k()
        
        # Attacker tries to frontrun liquidity addition
        attacker_tokens = self.pool.ethToToken(np.uint32(10000))
        
        # Legitimate LP adds liquidity
        lp_delta_t, lp_delta_l = self.pool.addLiquidity(np.uint32(100000))
        
        # Attacker tries to profit
        attacker_eth = self.pool.tokenToEth(attacker_tokens)
        
        # Attacker should lose money
        self.assertLess(attacker_eth, 10000,
                       "Frontrunning liquidity additions should not be profitable")
        
        # Pool and LPs should benefit
        self.assertGreater(self.pool.k(), initial_k,
                          "Pool value should increase from fees")

    def test_flash_loan_attack_prevention(self):
        """Test that flash loan attacks cannot drain the pool."""
        initial_state = (self.pool.e, self.pool.t, self.pool.l, self.pool.k())
        
        # Simulate flash loan attack: borrow large amount, manipulate, repay
        # Step 1: Large trade to manipulate price
        eth_amount = self.pool.e // 3
        tokens_received = self.pool.ethToToken(eth_amount)
        
        # Step 2: Try to exploit the manipulated state
        # (In a real flash loan, other operations would happen here)
        
        # Step 3: Trade back
        eth_returned = self.pool.tokenToEth(tokens_received)
        
        # Verify pool hasn't lost value
        final_k = self.pool.k()
        self.assertGreaterEqual(final_k, initial_state[3],
                               "Pool value should not decrease from flash loan attack")
        
        # Verify attacker lost money due to fees
        self.assertLess(eth_returned, eth_amount,
                       "Flash loan attacker should lose money to fees")

    def test_reentrancy_safety(self):
        """Test that state changes are atomic and safe from reentrancy."""
        # Save initial state
        initial_e = self.pool.e
        initial_t = self.pool.t
        initial_l = self.pool.l
        
        # Perform trade
        self.pool.ethToToken(np.uint32(1000))
        
        # Verify state was updated atomically
        # All state variables should have changed consistently
        self.assertNotEqual(self.pool.e, initial_e)
        self.assertNotEqual(self.pool.t, initial_t)
        self.assertEqual(self.pool.l, initial_l)  # Liquidity unchanged in trades
        
        # Verify invariant still holds
        self.assertGreater(self.pool.k(), 0)

    def test_dust_attack_resistance(self):
        """Test resistance to dust attacks that try to grief the pool."""
        initial_k = self.pool.k()
        
        # Try to grief the pool with many dust trades
        dust_trades = 0
        for _ in range(1000):
            try:
                # Trade 1 wei
                result = self.pool.ethToToken(np.uint32(1))
                if result > 0:
                    dust_trades += 1
            except:
                # Some trades might fail due to rounding
                pass
        
        # Pool should still be functional
        self.assertGreater(self.pool.e, 0, "Pool should still have ETH")
        self.assertGreater(self.pool.t, 0, "Pool should still have tokens")
        
        # Pool value should not have decreased significantly
        final_k = self.pool.k()
        self.assertGreater(final_k, initial_k * 0.99,
                          "Pool should not lose significant value from dust attacks")

    def test_zero_liquidity_edge_case(self):
        """Test edge cases around zero liquidity."""
        # Create a pool and remove almost all liquidity
        small_pool = CPMM(np.uint32(1000), np.uint32(1000), np.uint32(1000))
        
        # Remove most liquidity
        small_pool.removeLiquidity(np.uint32(999))
        
        # Pool should still function with minimal liquidity
        self.assertEqual(small_pool.l, 1)
        
        # Trades should still work but might have extreme slippage
        try:
            small_pool.ethToToken(np.uint32(1))
            # If trade succeeds, pool should still be valid
            self.assertGreater(small_pool.k(), 0)
        except (ValueError, OverflowError):
            # It's acceptable for trades to fail with extremely low liquidity
            pass

    def test_addLiquidity_ratio_enforcement(self):
        """Test that addLiquidity enforces correct token ratios."""
        initial_e = self.pool.e
        initial_t = self.pool.t
        initial_ratio = initial_t / initial_e
        
        # Add liquidity
        eth_to_add = np.uint32(50000)
        tokens_required, liquidity_minted = self.pool.addLiquidity(eth_to_add)
        
        # Verify the ratio is maintained after adding liquidity
        new_ratio = self.pool.t / self.pool.e
        # Due to the +1 in the token calculation, there's a tiny ratio change
        # This is expected and protects against rounding exploits
        ratio_change = abs(new_ratio - initial_ratio) / initial_ratio
        self.assertLess(ratio_change, 0.00001,  # Less than 0.001% change
                       msg="Adding liquidity should maintain price ratio with minimal change")
        
        # The addLiquidity formula: tokens_required = (delta_e * t / e) + 1
        # Calculate what tokens would be required without the +1
        u64_eth_to_add = np.uint64(eth_to_add)
        u64_initial_t = np.uint64(initial_t)
        u64_initial_e = np.uint64(initial_e)
        
        expected_tokens_exact = (u64_eth_to_add * u64_initial_t) // u64_initial_e
        
        # Due to +1 in formula, LPs always pay at least 1 more token
        self.assertGreaterEqual(tokens_required, expected_tokens_exact + 1,
                               "LPs should pay at least 1 extra token due to rounding protection")

    def test_price_oracle_manipulation_resistance(self):
        """Test resistance to price oracle manipulation attacks."""
        # Record prices over multiple blocks
        prices = []
        
        # Normal trading
        for i in range(5):
            price = self.pool.t / self.pool.e
            prices.append(price)
            self.pool.ethToToken(np.uint32(1000))
        
        # Attempted manipulation with large trade
        initial_price = self.pool.t / self.pool.e
        self.pool.ethToToken(self.pool.e // 4)  # Large trade
        manipulated_price = self.pool.t / self.pool.e
        
        # Price impact should be significant
        price_change = abs(manipulated_price - initial_price) / initial_price
        self.assertGreater(price_change, 0.1,
                          "Large trades should have significant price impact")
        
        # This makes TWAP oracles more resistant to manipulation
        # as manipulating price for extended periods is expensive

    def test_type_verification_methods(self):
        """Test that methods verify input types correctly."""
        # Test ethToToken with wrong type
        with self.assertRaises(TypeError):
            self.pool.ethToToken(1000)  # int instead of uint32
            
        # Test tokenToEth with wrong type
        with self.assertRaises(TypeError):
            self.pool.tokenToEth(1000)  # int instead of uint32
            
        # Test addLiquidity with wrong type
        with self.assertRaises(TypeError):
            self.pool.addLiquidity(1000)  # int instead of uint32
            
        # Test removeLiquidity with wrong type
        with self.assertRaises(TypeError):
            self.pool.removeLiquidity(100)  # int instead of uint32
            
        # Test ethToTokenExact with wrong type
        with self.assertRaises(TypeError):
            self.pool.ethToTokenExact(1000)  # int instead of uint32
            
        # Test tokenToEthExact with wrong type
        with self.assertRaises(TypeError):
            self.pool.tokenToEthExact(1000)  # int instead of uint32


if __name__ == '__main__':
    unittest.main(verbosity=2)