"""
Real Estate Market Analyzer with Orchestrator and EvaluatorOptimizerLLM Workflow
-------------------------------------------------------------------------------
An integrated real estate analysis tool using the orchestrator implementation
for property market analysis, investment evaluation, and neighborhood research.
Includes RentSpider API integration for enhanced property data.
"""

import asyncio
import os
import sys
import json
import aiohttp
import time
from datetime import datetime
from typing import Optional, Dict, Any
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.human_input.handler import console_input_callback
from mcp_agent.elicitation.handler import console_elicitation_callback
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Configuration values
OUTPUT_DIR = "property_reports"
LOCATION = "Austin, TX" if len(sys.argv) <= 1 else " ".join(sys.argv[1:])
PROPERTY_TYPE = "single family homes"  # Can be modified for apartments, condos, etc.
MAX_ITERATIONS = 3

# RentSpider API Configuration
RENTSPIDER_API_KEY = os.getenv("RENTSPIDER_API_KEY")  # Set this environment variable
RENTSPIDER_BASE_URL = "https://api.rentspider.com/v1"

# Initialize app with human input and elicitation callbacks
app = MCPApp(
    name="real_estate_analyzer",
    human_input_callback=console_input_callback,
    elicitation_callback=console_elicitation_callback,
)


class RentSpiderClient:
    """Client for interacting with RentSpider API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = RENTSPIDER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def search_properties(
        self, location: str, property_type: str = "all"
    ) -> Optional[Dict[Any, Any]]:
        """Search for properties in a specific location"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "location": location,
                    "property_type": property_type,
                    "limit": 50,
                }

                async with session.get(
                    f"{self.base_url}/properties/search",
                    headers=self.headers,
                    params=params,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"RentSpider API error: {response.status}")
                        return None
        except Exception as e:
            print(f"Error calling RentSpider API: {str(e)}")
            return None

    async def get_market_data(self, location: str) -> Optional[Dict[Any, Any]]:
        """Get market statistics for a location"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"location": location}

                async with session.get(
                    f"{self.base_url}/market/statistics",
                    headers=self.headers,
                    params=params,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"RentSpider market data error: {response.status}")
                        return None
        except Exception as e:
            print(f"Error getting market data: {str(e)}")
            return None

    async def get_rental_trends(self, location: str) -> Optional[Dict[Any, Any]]:
        """Get rental market trends for a location"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"location": location, "period": "12months"}

                async with session.get(
                    f"{self.base_url}/market/trends",
                    headers=self.headers,
                    params=params,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"RentSpider trends error: {response.status}")
                        return None
        except Exception as e:
            print(f"Error getting rental trends: {str(e)}")
            return None


async def main():
    # Create output directory and set up file paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{LOCATION.lower().replace(' ', '_').replace(',', '')}_property_report_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR, output_file)

    # Initialize RentSpider client
    rentspider_client = None
    if RENTSPIDER_API_KEY:
        rentspider_client = RentSpiderClient(RENTSPIDER_API_KEY)
        print("✅ RentSpider API initialized")
    else:
        print(
            "⚠️  RentSpider API key not found. Set RENTSPIDER_API_KEY environment variable for enhanced data."
        )

    async with app.run() as analyzer_app:
        context = analyzer_app.context
        logger = analyzer_app.logger

        # Configure filesystem server to use current directory
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        else:
            logger.warning("Filesystem server not configured - report saving may fail")

        # Check for g-search server
        if "g-search" not in context.config.mcp.servers:
            logger.warning(
                "Google Search server not found! This script requires g-search-mcp"
            )
            logger.info("You can install it with: npm install -g g-search-mcp")
            return False

        # --- COLLECT RENTSPIDER API DATA FIRST ---
        api_data = {}
        if rentspider_client:
            logger.info("Fetching data from RentSpider API...")

            # Get property listings
            properties = await rentspider_client.search_properties(
                LOCATION, PROPERTY_TYPE
            )
            if properties:
                api_data["properties"] = properties
                logger.info(
                    f"Retrieved {len(properties.get('results', []))} property listings"
                )

            # Get market statistics
            market_stats = await rentspider_client.get_market_data(LOCATION)
            if market_stats:
                api_data["market_stats"] = market_stats
                logger.info("Retrieved market statistics")

            # Get rental trends
            rental_trends = await rentspider_client.get_rental_trends(LOCATION)
            if rental_trends:
                api_data["rental_trends"] = rental_trends
                logger.info("Retrieved rental trends data")

        # --- DEFINE AGENTS ---

        # Market Research agent: Combines API data with web research and human input
        market_research_agent = Agent(
            name="market_researcher",
            instruction=f"""You are a world-class real estate market researcher specializing in {LOCATION}.

            IMPORTANT: You have access to RentSpider API data for {LOCATION}:
            {json.dumps(api_data, indent=2) if api_data else "No API data available - rely on web search"}

            You can also ask for human input when you need clarification or additional information.
            Use elicitation to ask the human for:
            - Specific property types they're interested in (if not clear)
            - Budget ranges for investment analysis
            - Investment timeline (short-term vs long-term)
            - Specific neighborhoods within {LOCATION} to focus on
            - Whether they're interested in rental properties vs personal residence

            If RentSpider API data is available, analyze it first for:
            - Property listings and pricing data
            - Market statistics and trends
            - Rental market information
            - Average prices per property type

            Then supplement with web search using these queries:
            1. "{LOCATION} real estate market trends 2025"
            2. "{LOCATION} {PROPERTY_TYPE} median home prices current"
            3. "{LOCATION} property market forecast 2025"
            4. "{LOCATION} days on market average 2025"
            5. "{LOCATION} real estate inventory levels"
            6. "Zillow {LOCATION} market data" OR "Realtor.com {LOCATION} trends"
            
            Combine API data with web research to extract:
            - Current median home prices and price trends (% change year-over-year)
            - Average rental rates and rental yields
            - Average days on market
            - Housing inventory levels (months of supply)
            - Market conditions (buyer's vs seller's market)
            - Recent sales volume and activity
            - Price per square foot data
            - Market forecasts and predictions
            
            If you need more specific information about the user's preferences or requirements,
            use elicitation to ask clarifying questions before proceeding with research.
            
            Prioritize RentSpider API data when available as it's more accurate.
            Be specific with numbers, dates, and cite all sources with URLs.
            """,
            server_names=["g-search", "fetch"],
        )

        # Research evaluator: Evaluates market research quality including API data
        market_research_evaluator = Agent(
            name="market_research_evaluator",
            instruction=f"""You are an expert real estate data evaluator specializing in market research quality.
            
            Evaluate the market research data for {LOCATION} based on these criteria:
            
            1. Data Accuracy: Are facts properly cited with source URLs? Are price figures and statistics precise?
               - Give extra credit if RentSpider API data is included (more reliable than web scraping)
               
            2. Data Completeness: Is all required information present?
               - Current median home prices with recent trends
               - Rental rates and rental yield data (important for investment analysis)
               - Days on market statistics
               - Market inventory levels
               - Price per square foot data
               - Market conditions assessment
               
            3. Data Recency: Is the data from the last 6 months? Are sources current and reliable?
               - RentSpider API data should be considered most current
               
            4. Source Quality: Are sources reputable?
               - RentSpider API data (highest quality)
               - Zillow, Realtor.com, MLS, local real estate reports
               - Multiple data sources for verification
            
            For each criterion, provide a rating:
            - EXCELLENT: Comprehensive data including API sources, highly reliable
            - GOOD: Most required data present from good sources
            - FAIR: Some data missing but basic metrics covered
            - POOR: Critical market data missing or outdated
            
            Provide specific feedback on what market data is missing or needs improvement.
            If median home prices, rental data, market trends, or days on market data are missing, 
            the overall rating should not exceed FAIR.""",
        )

        # Create the market research EvaluatorOptimizerLLM component
        market_research_controller = EvaluatorOptimizerLLM(
            optimizer=market_research_agent,
            evaluator=market_research_evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.EXCELLENT,
        )

        # Neighborhood Research agent: Focuses on local amenities with human input
        neighborhood_agent = Agent(
            name="neighborhood_researcher",
            instruction=f"""You are a world-class neighborhood research specialist for {LOCATION}.
            
            You can ask for human input to better understand their priorities and needs.
            Use elicitation to ask about:
            - What neighborhood factors are most important to them (schools, safety, nightlife, etc.)
            - Whether they have children (affects school importance)
            - Transportation needs (car vs public transit)
            - Lifestyle preferences (urban vs suburban feel)
            - Specific concerns about the area
            
            Based on their input and general research, gather information about:
            1. "{LOCATION} neighborhood amenities schools"
            2. "{LOCATION} crime statistics safety ratings"
            3. "{LOCATION} walkability transportation public transit"
            4. "{LOCATION} local economy job market employment"
            5. "{LOCATION} demographics population income"
            6. "{LOCATION} future development plans infrastructure"
            
            Focus on collecting data about:
            - School ratings and educational quality
            - Crime rates and safety metrics
            - Walkability scores and transportation options
            - Local employment and economic indicators
            - Demographics and income levels
            - Planned developments and infrastructure projects
            - Local amenities (parks, shopping, restaurants)
            - Property tax rates and municipal services
            
            Tailor your research focus based on what the human indicates is most important to them.
            Cite all sources and be specific with ratings, scores, and statistics.""",
            server_names=["g-search", "fetch"],
        )

        # Investment Analysis agent: Analyzes investment potential with personalized input
        investment_analyst = Agent(
            name="investment_analyst",
            instruction=f"""You are a world-class real estate investment analyst for {LOCATION}.
            
            You can ask for human input to provide personalized investment advice.
            Use elicitation to ask about:
            - Investment budget and down payment available
            - Investment timeline (how long they plan to hold)
            - Risk tolerance (conservative vs aggressive)
            - Investment goals (cash flow vs appreciation)
            - Experience level with real estate investing
            - Whether they want rental properties or fix-and-flip
            - Preferred property management approach (self vs hired)
            
            Analyze the market and neighborhood data to provide investment insights:
            
            1. Market Conditions Analysis:
               - Is it currently a buyer's or seller's market?
               - How do current prices compare to historical trends?
               - What does inventory levels suggest about competition?
            
            2. Rental Market Analysis (prioritize RentSpider API data if available):
               - Current rental rates for different property types
               - Rental yield calculations (annual rent / purchase price)
               - Rental market trends and demand
               - Cash flow potential for investment properties
            
            3. Investment Timing Assessment:
               - Are prices trending up or down?
               - How long are properties staying on market?
               - What do forecasts suggest for the next 12-24 months?
               - Rental rate trends and future projections
            
            4. Personalized Investment Strategy:
               - Recommendations based on their budget and goals
               - Property types that match their criteria
               - Neighborhoods that align with their strategy
               - Entry and exit strategies
            
            5. Risk Assessment:
               - Market volatility indicators
               - Economic factors that could affect values
               - Supply and demand imbalances
               - Rental market stability
            
            6. Investment Recommendations:
               - Overall investment attractiveness (1-10 scale)
               - Best property types for their specific situation
               - Optimal buying strategy and timing
               - Expected ROI potential including rental income
               - Cash-on-cash return estimates
               - Financing recommendations
            
            Tailor all recommendations to the human's specific situation, budget, and goals.
            Be specific with numbers, percentages, and rental yield calculations.
            Provide clear reasoning for all assessments.""",
            server_names=["fetch"],
        )

        # Report writer: Creates comprehensive personalized real estate report
        report_writer = Agent(
            name="real_estate_report_writer",
            instruction=f"""Create a comprehensive and personalized real estate market analysis report for {LOCATION}.
            
            Before writing the report, you may want to ask for final clarifications:
            - Report format preferences (detailed vs summary)
            - Any specific sections they want emphasized
            - Whether they want action items/next steps included
            - Timeline for making investment decisions
            
            Structure the report with these sections:
            
            1. **Executive Summary**
               - Overall market assessment and personalized investment recommendation
               - Key findings tailored to their situation in 3-4 bullet points
               - Investment attractiveness rating (1-10) with reasoning
            
            2. **Market Overview**
               - Current median home prices and recent trends
               - Days on market and inventory levels
               - Market conditions (buyer's vs seller's market)
               - Price per square foot data
            
            3. **Rental Market Analysis** (if RentSpider API data available)
               - Current rental rates by property type
               - Rental yield analysis and cash flow potential
               - Rental market trends and vacancy rates
               - Rental vs purchase price ratios
            
            4. **Market Trends & Forecasts**
               - Year-over-year price changes
               - Market predictions for next 12-24 months
               - Supply and demand analysis
               - Rental rate forecasts
            
            5. **Neighborhood Analysis**
               - Factors most relevant to the user's priorities
               - School ratings and educational quality (if important to them)
               - Safety and crime statistics
               - Walkability and transportation
               - Local amenities and quality of life factors
            
            6. **Personalized Investment Analysis**
               - Investment attractiveness rating (1-10) for their situation
               - Risk assessment based on their tolerance
               - ROI potential including rental income
               - Cash-on-cash return estimates for their budget
               - Recommended investment strategies
               - Property types that match their criteria
            
            7. **Action Plan & Next Steps**
               - Specific recommendations for their situation
               - Timeline for decision making
               - Key metrics to monitor
               - Suggested next actions
            
            8. **Demographics & Economics**
               - Population and income trends
               - Employment and economic indicators
               - Future development plans
            
            9. **Data Sources & References**
               - RentSpider API data (if used)
               - Web research sources with URLs
               - Data reliability notes
               - Disclaimer about market volatility
            
            Format as clean markdown with appropriate headers, tables, and bullet points.
            Include exact figures with proper formatting (e.g., $XXX,XXX, XX.X%).
            Highlight personalized recommendations and action items prominently.
            Keep under 2000 words total but ensure comprehensive coverage.
            
            Save the report to "{output_path}".""",
            server_names=["filesystem"],
        )

        # --- CREATE THE ORCHESTRATOR ---
        logger.info(f"Initializing real estate analysis workflow for {LOCATION}")

        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                market_research_controller,
                neighborhood_agent,
                investment_analyst,
                report_writer,
            ],
            plan_type="full",
        )

        # Define the task for the orchestrator
        task = f"""Create a comprehensive and personalized real estate market analysis report for {LOCATION} by following these steps:

        1. Use the 'market_research_controller' (EvaluatorOptimizerLLM component) to gather 
           high-quality market data for {LOCATION}. This component will automatically evaluate 
           and improve the research until it reaches EXCELLENT quality.
           
           The component has access to RentSpider API data: {bool(api_data)}
           The agent can ask for human input to clarify:
           - Specific property types of interest
           - Budget ranges for analysis
           - Investment vs personal residence focus
           
           Required market data:
           - Current median home prices and price trends
           - Rental rates and rental market analysis
           - Days on market and inventory levels
           - Market forecasts and conditions
           - Price per square foot data

        2. Use the 'neighborhood_researcher' to gather comprehensive neighborhood information:
           - Ask about user priorities (schools, safety, transportation, etc.)
           - School ratings and educational quality
           - Crime statistics and safety
           - Demographics and local economy
           - Amenities and quality of life factors

        3. Use the 'investment_analyst' to analyze all collected data and provide:
           - Ask about investment goals, budget, timeline, and risk tolerance
           - Personalized investment attractiveness assessment with rental yield analysis
           - Risk analysis and market timing recommendations tailored to user
           - ROI potential including rental income calculations for their budget
           - Cash flow analysis for investment properties matching their criteria

        4. Use the 'real_estate_report_writer' to create a comprehensive market report 
           and save it to: "{output_path}"
           - Ask about report format preferences and key focus areas
           - Include personalized recommendations and action plan

        Throughout the process, agents should use elicitation to ask clarifying questions
        that will help provide more personalized and actionable recommendations.
        
        The final report should be professional, data-driven, personalized to the user's
        specific situation and goals, and provide actionable investment insights for 
        {LOCATION} real estate market, with special emphasis on rental market analysis 
        when RentSpider API data is available."""

        # Run the orchestrator
        logger.info("Starting the interactive real estate analysis workflow")
        print("\n🎯 This analysis will be personalized to your needs.")
        print(
            "💬 You may be asked questions during the process to provide better recommendations.\n"
        )

        start_time = time.time()

        try:
            await orchestrator.generate_str(
                message=task, request_params=RequestParams(model="gpt-4o")
            )

            # Check if report was successfully created
            if os.path.exists(output_path):
                end_time = time.time()
                total_time = end_time - start_time
                logger.info("Report successfully generated: {output_path}")
                print("\n✅ Report generated successfully!")
                print(f"📁 Location: {output_path}")
                print(f"🏠 Market analyzed: {LOCATION}")
                print(f"⏱️  Total analysis time: {total_time:.2f}s")
                return True
            else:
                logger.error(f"Failed to create report at {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}")
            return False


if __name__ == "__main__":
    # Check for RentSpider API key
    if not RENTSPIDER_API_KEY:
        print("⚠️  RentSpider API key not found!")
        print("💡 Set environment variable: export RENTSPIDER_API_KEY='your-api-key'")
        print("🔗 Get your API key at: https://rentspider.com/api")
        print("📝 Analysis will continue with web search only...\n")

    if len(sys.argv) > 1:
        print(f"🏡 Analyzing real estate market for: {' '.join(sys.argv[1:])}")
    else:
        print(f"🏡 Analyzing real estate market for: {LOCATION} (default)")

    print("🤖 Interactive Real Estate Analysis with Elicitation")
    print("💬 You'll be asked questions to personalize your report")
    print("⏳ Starting analysis...\n")

    start = time.time()
    success = asyncio.run(main())
    end = time.time()
    total_time = end - start

    if success:
        print(f"\n🎉 Real estate analysis completed successfully in {total_time:.2f}s!")
        print("📊 Check the generated report for detailed market insights.")
        if RENTSPIDER_API_KEY:
            print("✅ Enhanced with RentSpider API data for rental market analysis")
        print("🎯 Report personalized based on your input")
    else:
        print(f"\n❌ Analysis failed after {total_time:.2f}s. Check logs for details.")
