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
        self.lead = 0
        self.time = 0.0
        self.has_ball = None

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
        self.inventory = {}

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

        self.tick = ticker
        self.p = price
        self.quant = quantity
        self.last_trade = [self.tick, self.p, self.quant]
        self.trade_exists = True

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
        
        if side == Side.BUY:
            self.orderbook['BUY'][ticker][price] += quantity
        
        else:
            self.orderbook['SELL'][ticker][price] += quantity

        if self.trade_exists:
            self.orderbook['BUY'][self.last_trade[0]][self.last_trade[1]] -= self.last_trade[2]
            self.orderbook['SELL'][self.last_trade[0]][self.last_trade[1]] -= self.last_trade[2]
            self.trade_exists = False


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
        if side == Side.BUY:
            self.inventory[ticker] += quantity
        else:
            self.inventory[ticker] -= quantity 

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
        self.check_order_book(Ticker(0))    

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
                disp: float = 10,
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
            lambda_h = np.random.gamma(shape = disp, scale = (h_fut / disp))
            lambda_a = np.random.gamma(shape = disp, scale =  (a_fut / disp))
        
            pts_h = np.random.poisson(lambda_h)
            pts_a = np.random.poisson(lambda_a)
        
            margin = lead + (pts_h - pts_a)
        
            wins = (margin > 0).sum()
            prob = wins / sim
        
            price = prob * 100
        
            return price
        return None
    


    def check_order_book(self, ticker: Ticker):

        self.fair_price = self.mc_prob(self.lead, self.time, self.spp, self.h_ppp, self.a_ppp, ball = self.has_ball)

        if self.fair_price is not None:
            sells = sorted(self.orderbook['SELL'][ticker].keys())
            buys = sorted(self.orderbook['BUY'][ticker].keys(), reverse=True)
            quantity = 10000 // self.fair_price

            for buy in buys:
                if self.fair_price + 3 < buy:
                    if quantity > self.orderbook['BUY'][ticker][buy]:
                        place_limit_order(Side(1), Ticker(0), self.orderbook['BUY'][ticker][buy], buy, True)
                        quantity -= self.orderbook['BUY'][ticker][buy]

                    else:
                        place_limit_order(Side(1), Ticker(0), quantity, buy, True)
                        break
        
            for sell in sells:
                if self.fair_price - 3 > sell:
                    if quantity > self.orderbook['SELL'][ticker][sell]:
                        place_limit_order(Side(0), Ticker(0), self.orderbook['SELL'][ticker][sell], sell, True)
                        quantity -= self.orderbook['SELL'][ticker][sell]

                    else:
                        place_limit_order(Side(0), Ticker(0), quantity, sell, True)
                        break

            place_limit_order(Side(1), Ticker(0), self.inventory[ticker], self.fair_price + 2, False)    
            
            
    
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




