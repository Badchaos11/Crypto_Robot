from binance import Client

api_key = 'unN9xy5u4RpltGH3Her5eV3mXWU1oJP86HM6y8xGMkFVBL6sUVEiIiEv0bUMV4eN'
api_secret = 'NRKyABoXEbeamGXpuS7QEiJgAAlsN2AjCatQhwVY2HnKIWoowTlw0Gbt4xAgSSMH'

client = Client(api_key=api_key, api_secret=api_secret, tld='com', testnet=True)
