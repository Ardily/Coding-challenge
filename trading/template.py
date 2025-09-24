"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional
import numpy as np
from collections import defaultdict

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        self.swaps = 0
        self.lead = 0
        self.time = 0
        self.hn_p = 0
        self.an_p = 0
        self.spp = 15.0
        self.period_start = 2880
        self.inventory = 0
        self.h_ppp = 1.1
        self.a_ppp = 1.1
        self.order_id = False
        self.output = True
        self.has_ball = None
        self.bids = {}
        self.asks = {}
        self.close_postion = False
        self.track_orders = []

        pass

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.orderbook = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.trade_exists = False
        self.reset_state()
        self.swaps = 0
        self.lead = 0
        self.hn_p = 0
        self.an_p = 0
        self.spp = 15.0
        self.period_start = 2880
        self.inventory = 0
        self.h_ppp = 1.1
        self.a_ppp = 1.1
        self.order_id = False
        self.output = True
        self.has_ball = None
        self.bids = {}
        self.asks = {}
        self.close_postion = False
        self.track_orders = []


    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """

        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        book = self.bids if side == Side.BUY else self.asks
        if quantity > 0:
            book[price] = quantity
        elif price in book:
            del book[price]


    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Called periodically with a complete snapshot of the orderbook.

        This provides the full current state of all bids and asks, useful for 
        verification and algorithms that need the complete market picture.

        Parameters
        ----------
        ticker
            Ticker of the orderbook snapshot (Ticker.TEAM_A)
        bids
            List of (price, quantity) tuples for all current bids, sorted by price descending
        asks  
            List of (price, quantity) tuples for all current asks, sorted by price ascending
        """
        self.bids.clear()
        self.asks.clear()

        for p, q in bids:
            self.bids[p] = q
        for p, q in asks:
            self.asks[p] = q
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """

        multiplier = 1 if side == Side.BUY else -1
        self.inventory += multiplier * quantity

        self.track_orders.append((side, price, quantity))

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """
        self.swaps = self.possession(event_type, home_away, rebound_type, self.swaps)
        self.lead = home_score - away_score
        self.time = time_seconds

        if event_type == 'JUMP_BALL' and time_seconds in (2880, 2400):
            start_time = time_seconds
        else:
            start_time = 2640
        if (home_score + away_score) > 20 and self.swaps > 20 and self.hn_p !=0 and self.an_p != 0:
            self.spp = (start_time - time_seconds) / self.swaps
            if self.spp < 0:
                self.spp = (2880 - time_seconds) / self.swaps
            self.h_ppp = home_score / self.hn_p
            self.a_ppp = away_score / self.an_p

        print(f"{event_type} {home_score} - {away_score}")

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.reset_state()
            return
        
        if start_time - self.time > 60 and self.close_postion == False:
            self.check_order_book(Ticker.TEAM_A)

        if self.time < 30:
            if self.inventory < 0:
                place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.inventory))
                self.close_postion == True
            
            elif self.inventory > 0:
                place_market_order(Side.SELL, Ticker.TEAM_A, abs(self.inventory))
                self.close_postion == True

            else:
                self.close_postion == True

    def inc_position(self, team):
        if team == 'home':
            self.hn_p += 1
        elif team == 'away':
            self.an_p += 1

    def mc_prob(self,
                lead: int,
                time: float,
                spp: float = 15,
                h_ppp: float = 1,
                a_ppp: float = 1,
                disp: float = 12,
                sim: int = 50000,
                ball: str | None = None):
        
        pos_left = max(0, time) / max(4, spp)
        N = np.random.poisson(pos_left, sim)

        if ball == 'home':
            n_h = (N + 1) // 2
            n_a = N - n_h
        elif ball == 'away':
            n_a = (N + 1) // 2
            n_h = N - n_a
        else:
            n_h = N // 2
            n_a = N - n_h
        
        h_fut = n_h * h_ppp
        a_fut = n_a * a_ppp

        if disp > 0.1:
            lambda_h = np.random.gamma(shape = disp, scale = (h_fut / disp), size = sim)
            lambda_a = np.random.gamma(shape = disp, scale =  (a_fut / disp), size = sim)
        
            pts_h = np.random.poisson(lambda_h)
            pts_a = np.random.poisson(lambda_a)
        
            margin = lead + (pts_h - pts_a)
        
            wins = (margin > 0).sum()
            ties = (margin == 0).sum()
            prob = (wins + 0.5 * ties) / sim
        
            price = prob * 100
        
            return price
        return None
    


    def check_order_book(self, ticker: Ticker):

        self.fair_price = self.mc_prob(self.lead, self.time, self.spp, self.h_ppp, self.a_ppp, ball = self.has_ball)
        edge = 3
        if self.fair_price is not None:
            sells = sorted(self.asks.items(), key= lambda x: x[0])
            buys = sorted(self.bids.items(), key = lambda x: x[0], reverse=True)
            print(f'fair price: {self.fair_price}, market mid: {(max(sell[0][0])+min(buys[0][0])) / 2}')
            quantity = 5000 // self.fair_price

            inv = self.inventory
            for buy, q in buys:
                if inv < -700:
                    continue

                else:
                    if self.fair_price + edge < buy:
                        if quantity > q:
                            place_limit_order(Side.SELL, Ticker.TEAM_A, q, buy, True)
                            quantity -= q

                        else:
                            place_limit_order(Side.SELL, Ticker.TEAM_A, quantity, buy, True)
                            break
        
            for sell, q in sells:
                if inv > 700:
                    continue

                else:
                    if self.fair_price - edge > sell:
                        if quantity > q:
                            place_limit_order(Side.BUY, Ticker.TEAM_A, q, sell, True)
                            quantity -= q

                        else:
                            place_limit_order(Side.BUY, Ticker.TEAM_A, quantity, sell, True)
                            break

            place_limit_order(Side.SELL, Ticker.TEAM_A, inv, self.fair_price + edge)
                
    
    def possession(self, event_type, team, rebound, swaps):
        if team not in ('home', 'away'):
            return swaps
        
        if event_type in ('SCORE', 'TURNOVER'):
            self.has_ball = 'away' if team == 'home' else 'home'
            swaps += 1
            self.inc_position(self.has_ball)

        elif event_type == 'STEAL':
            self.has_ball = team
            swaps += 1
            self.inc_position(self.has_ball)
        elif event_type == 'REBOUND':
            if rebound == 'DEFENSIVE':
                self.has_ball = team
                swaps += 1
                self.inc_position(self.has_ball)
        
        elif event_type == 'JUMP_BALL':
            self.has_ball = team
            swaps += 1
            self.inc_position(self.has_ball)
        return swaps




