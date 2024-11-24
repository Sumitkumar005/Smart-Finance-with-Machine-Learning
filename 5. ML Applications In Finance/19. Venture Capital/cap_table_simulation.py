# Create a tool that allows for the simulation of various funding rounds and exits, showing how ownership and dilution
# evolve over time.
import pandas as pd

# Initialize the company details and funding round information
class FundingSimulation:
    def __init__(self, founder_name, total_shares):
        self.total_shares = total_shares
        self.shares = pd.DataFrame(columns=['Investor', 'Shares', 'Ownership %'])
        
        # Initial shares for the founder (100% ownership)
        self.shares = self.shares.append({'Investor': founder_name, 'Shares': total_shares, 'Ownership %': 100.0}, ignore_index=True)
        
        self.funding_rounds = []
        self.exit_value = None

    def add_funding_round(self, round_name, investor_name, investment_amount, company_valuation):
        """ Add a new funding round to the simulation """
        
        # New investor's shares = Investment amount / company valuation * total shares
        new_shares = (investment_amount / company_valuation) * self.total_shares
        
        # Add the new investor's details
        self.shares = self.shares.append({'Investor': investor_name, 'Shares': new_shares, 'Ownership %': 0}, ignore_index=True)
        
        # Update ownership percentage for all investors
        self.update_ownership()
        
        self.funding_rounds.append({
            'Round': round_name,
            'Investor': investor_name,
            'Investment Amount': investment_amount,
            'Shares Issued': new_shares,
            'Post-Money Valuation': company_valuation,
            'Dilution': (new_shares / self.total_shares) * 100
        })

    def update_ownership(self):
        """ Update the ownership percentage for all investors after a funding round """
        total_shares = self.shares['Shares'].sum()
        self.shares['Ownership %'] = (self.shares['Shares'] / total_shares) * 100

    def simulate_exit(self, exit_value):
        """ Simulate an exit event (e.g., IPO or acquisition) """
        self.exit_value = exit_value
        total_value = exit_value * self.shares['Shares'].sum() / self.total_shares
        return total_value

    def print_ownership(self):
        """ Print the current ownership distribution """
        print(self.shares)
    
    def print_funding_rounds(self):
        """ Print a summary of all funding rounds """
        for round_info in self.funding_rounds:
            print(round_info)

    def print_exit(self):
        """ Print the results of the exit event """
        if self.exit_value is not None:
            print(f"Exit Value: ${self.exit_value}")
            print(f"Total company valuation at exit: ${self.exit_value * self.total_shares}")
            print(f"Investor distribution at exit:")
            print(self.shares)
        else:
            print("No exit event simulated yet.")

# Example of using the FundingSimulation class

# Create a company with 1 million shares (100% owned by the founder initially)
company = FundingSimulation("Founder", 1000000)

# Adding funding rounds
company.add_funding_round("Seed Round", "Angel Investor", 500000, 5000000)  # Angel invests $500,000 at $5M valuation
company.add_funding_round("Series A", "VC Firm", 2000000, 10000000)  # VC Firm invests $2M at $10M valuation

# Print ownership after funding rounds
print("\n--- Ownership After Funding Rounds ---")
company.print_ownership()

# Simulate an exit (e.g., an acquisition of the company at a $50M valuation)
exit_value = 50_000_000  # Exit value of $50 million
exit_total_value = company.simulate_exit(exit_value)

# Print exit scenario
print("\n--- Exit Simulation ---")
company.print_exit()

# Print all funding rounds
print("\n--- Funding Rounds Summary ---")
company.print_funding_rounds()

