def calculate_position_size(portfolio_value, risk_per_trade, stop_loss_percent):
    max_loss = portfolio_value * risk_per_trade
    position_size = max_loss / stop_loss_percent
    return position_size

def trailing_stop_loss(data, entry_price, stop_loss_percent, trailing_percent):
    current_price = data['Close'].iloc[-1]
    initial_stop_loss = entry_price * (1 - stop_loss_percent)
    trailing_stop = entry_price * (1 - trailing_percent)

    if current_price > entry_price:
        new_trailing_stop = current_price * (1 - trailing_percent)
        if new_trailing_stop > trailing_stop:
            trailing_stop = new_trailing_stop

    return max(trailing_stop, initial_stop_loss)