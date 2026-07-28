"""
DeepEval Test Suite for Feedback-to-Engineering-Insights Pipeline
==================================================================
Run with: python test_feedback_engineering.py

Prerequisites:
  pip install deepeval openai
  Set OPENAI_API_KEY as environment variable (do NOT hardcode)
    Windows PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."
    Mac/Linux:           export OPENAI_API_KEY="sk-proj-..."
"""
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Set OPENAI_API_KEY environment variable first."); sys.exit(1)

TEST_FEEDBACK = {
    "F1_bug_checkout": {"input": "The checkout page keeps freezing when I try to apply a discount code on my iPhone 14. Happens every time. I've lost 3 orders this week because of it. Using Safari, latest iOS. -- App Store review, 1 star", "label": "detailed"},
    "F2_perf_api": {"input": "The API response times have degraded significantly since last month. Our p95 latency went from 200ms to 800ms. We're on the Enterprise plan and this is affecting our production systems. We've opened 3 tickets about this already. Account: api-team@bigcorp.com -- Support ticket", "label": "detailed"},
    "F3_usability_onboarding": {"input": "The onboarding flow is confusing. I signed up yesterday and still can't figure out how to create my first project. The getting started guide links to a 404 page. I watched 3 YouTube tutorials and they all show a different UI than what I see. Free plan, Chrome on Windows. -- Support ticket", "label": "detailed"},
    "F4_security": {"input": "Login alert at 3:47 AM EST from IP 203.0.113.42 (Romania). I did NOT log in. Account: lisa.t@company.com, #AC-41209. Changed password, enabled 2FA. Need: 1) Was data accessed? 2) 30-day audit log 3) Other accounts affected? -- Support ticket", "label": "detailed"},
    "F5_vague_angry": {"input": "This app is garbage. Total waste of money.", "label": "vague"},
    "F6_vague_minimal": {"input": "Terrible. Just terrible. 0 stars if I could.", "label": "vague"},
    "F7_vague_indifferent": {"input": "Meh. It's fine I guess.", "label": "vague"},
    "F8_vague_demand": {"input": "Fix your stuff.", "label": "vague"},
}

# ============================================================
# PASTE YOUR N8N PIPELINE OUTPUTS HERE
# ============================================================
# PIPELINE_OUTPUTS = {
#      "F1_bug_checkout": """=== ENGINEERING INSIGHT === TITLE: Checkout freeze on iOS Safari when applying discount codes TICKET_TYPE: BUG SEVERITY: HIGH TECHNICAL SUMMARY: The checkout page freezes when attempting to apply a discount code on an iPhone 14 running latest iOS version, specifically on Safari. This issue occurs consistently, resulting in lost orders. The root cause of the freeze is unknown.

# AFFECTED COMPONENTS:

# System: Checkout
# Platform: iOS
# Environment: Mobile
# REPRODUCTION STEPS:

# Open the app on an iPhone 14 with latest iOS version
# Navigate to the checkout page
# Attempt to apply a discount code
# INVESTIGATION CHECKLIST:

# [ ] Review Safari browser version and compatibility with the app
# [ ] Inspect the checkout page JavaScript for any potential freezes or errors
# [ ] Verify the discount code API integration for any issues
# DATA GAPS FOR ENGINEERING:

# Error logs or crash reports from the iOS app
# Details about the specific Safari browser version and its configuration
# Additional information about the discount code API integration
# PRIORITY RECOMMENDATION: P1-CRITICAL PRIORITY JUSTIFICATION: The issue results in lost orders, which has a significant impact on user experience and potentially revenue.""",

#      "F2_perf_api": """=== ENGINEERING INSIGHT === TITLE: API Response Times Degradation on Enterprise Plan TICKET_TYPE: PERFORMANCE SEVERITY: P1-CRITICAL TECHNICAL SUMMARY: accustomed API response times have increased significantly, with p95 latency rising from 200ms to 800ms, affecting production systems on the Enterpriseㅋㅋ plan. This degradation has been persistent since last month. Further investigation is required to determine the root cause.

# AFFECTED COMPONENTS:

# System: API
# آسیاب Platform: Enterprise
# Environment: Production
# REPRODUCTION STEPS:

# The issue has been reported across multiple tickets, but no specific steps to reproduce the issue have been provided.
# INVESTIGATION CHECKLIST:

# [ ] Review system logs for any changes or anomalies in the last month
# [ ] Check for any recent changes to API configuration, deployment, or scaling
# [ ] Verify resource utilization and performance metrics for the API
# DATA GAPS FOR ENGINEERING:

# Detailed performance metrics and logs for the last month
# Information on recent changes to the API, deployment, or scaling
# PRIORITY RECOMMENDATION: P1-CRITICAL PRIORITY JUSTIFICATION: The significant degradation in API response times has a blast radius of impacting all users on the Enterprise plan, which could lead to data integrity risks and overall system instability.""",

#      "F3_usability_onboarding": """=== ENGINEERING INSIGHT === TITLE: Confusing onboarding flow and 404 error on getting started guide TICKET_TYPE: BUG SEVERITY: HIGH TECHNICAL SUMMARY: The onboarding flow is causing user confusion due to a discrepancy between the getting started guide and the actual UI. This results in users being unable to create their first project. The issue is further complicated by outdated YouTube tutorials that show a different UI.

# AFFECTED COMPONENTS:

# System: Onboarding
# Platform: Web (Chrome on Windows)
# Environment: Free plan
# REPRODUCTION STEPS:

# Signed up for a free plan yesterday
# Attempted to access the getting started guide
# Encountered a 404 error
# Watched 3 YouTube tutorials that showed a different UI than the actual application
# Still unable to create a project
# INVESTIGATION CHECKLIST:

# Verify the getting started guide link is correct and not pointing to a 404 page
# Review the current UI to identify any changes that may have occurred recently
# Investigate the source of the outdated YouTube tutorials and determine if they can be updated
# Check if there are any known issues with the onboarding flow for free plan users
# DATA GAPS FOR ENGINEERING:

# Technical details about the 404 error (error code, log entries, etc.)
# Information about recent changes to the onboarding flow or UI
# PRIORITY RECOMMENDATION: P2-HIGH PRIORITY JUSTIFICATION: This issue is impacting new users and preventing them from creating their first project, which is a critical step in the onboarding process. The high blast radius and potential for user frustration justify a high priority.""",

#      "F4_security": """=== ENGINEERING INSIGHT === TITLE: Unauthorized login attempt from Romania at 3:47 AM EST TICKET_TYPE: INVESTIGATION_NEEDED SEVERITY: HIGH

# TECHNICAL SUMMARY: Insufficient feedback for engineering action. An unauthorized login attempt was made from IP 203.getZero.113.42 in Romania at 3:47 AM EST to account lisa.t@company.com (#AC-41209), which has a changed password and enabled 2FA. Engineering needs more information to determine the scope of the incident.

# AFFECTED COMPONENTS:

# System: Authentication
# Platform: Web
# Environment: Production
# REPRODUCTION STEPS:

# The user reported an unauthorized login attempt to their account lisa.t@company.com (#AC-41209) from IP 203.0.113.42 in Romania at 3:47 AM EST.
# INVESTIGATION CHECKLIST:

# [ ] Reach out to user for technical details (e.g., what actions they took before the incident, any other accounts they suspect may be affected)
# [ ] Review the 30-day audit log for the account to determine if any unauthorized access occurred
# [ ] Check if the IP 203.0.113.42 has been used for any other suspicious activity in our system
# DATA GAPS FOR ENGINEERING:

# Details about the user's actions before the incident
# Information about other accounts that may be affected
# Any other relevant technical details from the user
# PRIORITY RECOMMENDATION: P1-CRITICAL PRIORITY JUSTIFICATION: High priority due to potential data breach and security risk, as the user's password has been changed and 2FA is enabled, but the login attempt still occurred.""",

#      "F5_vague_angry": """=== ENGINEERING INSIGHT === TITLE: Unclear technical issue with the app TICKET_TYPE: INVESTIGATION_NEEDED SEVERITY: UNCLASSIFIED

# TECHNICAL SUMMARY: Insufficient feedback for engineering action. The user has reported dissatisfaction with the app, but no specific technical issue is described.

# AFFECTED COMPONENTS:

# System: Entire app
# Platform: Unknown
# Environment: Unknown
# REPRODUCTION STEPS:

# The user reports that the app is "garbage" and a "total waste of money",(reordered to focus on what they did, but the user didn't describe any steps)
# INVESTIGATION CHECKLIST:

# Reach out to user for technical details
# Clarify what specific issues they are experiencing
# DATA GAPS FOR ENGINEERING:

# Specific technical issue description
# Platform and environment details
# PRIORITY RECOMMENDATION: Unknown PRIORITY JUSTIFICATION: Insufficient information to determine priority.""",

#      "F6_vague_minimal": """=== ENGINEERING INSIGHT === TITLE: Poor user experience with checkout process TICKET_TYPE: BUG SEVERITY: P2-HIGH TECHNICAL SUMMARY: The checkout process is not functioning as expected, resulting in a poor user experience. This issue is specific to iOS Safari and occurs when applying discount codes. The exact nature of the failure is unknown.

# AFFECTED COMPONENTS:

# System: Checkout
# Platform: iOS
# Environment: Safari browser
# REPRODUCTION STEPS:

# Users report that the checkout process freezes when attempting to apply discount codes on iOS Safari.
# INVESTIGATION CHECKLIST:

# [ ] Review checkout API calls to identify any potential issues
# [ ] Inspect Safari-specific rendering and JavaScript execution
# [ ] Verify discount code validation and application logic
# DATA GAPS FOR ENGINEERING:

# More detailed error logs or crash reports from affected users
# Information on the specific version of iOS and Safari being used
# PRIORITY RECOMMENDATION: P2-HIGH PRIORITY JUSTIFICATION: This issue is impacting the user experience and could potentially result in lost sales, although the exact scale is unknown. Further investigation is required to understand the root cause and implement a fix.""",

#      "F7_vague_indifferent": """=== ENGINEERING INSIGHT === TITLE: Insufficient feedback for engineering action TICKET_TYPE: INVESTIGATION_NEEDED SEVERITY: UNKNOWN TECHNICAL SUMMARY: Insufficient feedback for engineering action. The user has provided feedback but it does not contain enough information for the engineering team to understand the MPU issue.

# AFFECTED COMPONENTS:

# System: Unknown
# Platform: Unknown
# Environment: Unknown
# REPRODUCTION STEPS:

# User reported the issue but did not provide enough information.
# INVESTIGATION CHECKLIST:

# Reach out to user for technical details -biggest issue is lack of information from user
# possibly need to clarify what the user means by 'fine, I guess'
# DATA GAPS FOR ENGINEERING:

# Technical information about the issue
# Specific steps to reproduce the problem
# PRIORITY RECOMMENDATION: P4-LOW PRIORITY JUSTIFICATION: The issue is unclear and there is no immediate impact on users, but further investigation is needed to determine the nature of the issue.""",

#      "F8_vague_demand": """I need more information from you to create a well-structured engineering insight. Please provide the feedback from the user, and the classifier output, which should include the following fields:

# TICKET_TYPE (BUG / FEATURE / PERFORMANCE / INVESTIGATION_NEEDED)
# SEVERITY (from classifier)
# AFFECTED_SYSTEM
# PLATFORM
# ENVIRONMENT
# Please provide the feedback from the user and the classifierSupportActionBar""",

#  }

# sample outputs for testing the evaluation framework using F1-F8 test use-cases where output is collected using basemodel (before fine-tuning).
#PIPELINE_OUTPUTS = {
#    "F1_bug_checkout": """=== ENGINEERING INSIGHT ===\nTITLE: Checkout freeze on iOS Safari when applying discount codes\nTICKET_TYPE: BUG\nSEVERITY: HIGH\n\nTECHNICAL SUMMARY: The checkout page freezes when attempting to apply a discount code on an iPhone 14 running the latest iOS version, specifically using Safari. This issue occurs consistently and has resulted in lost orders. The exact cause of the freeze is unknown.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: iOS\n- Environment: Mobile\n\nREPRODUCTION STEPS:\n1. Open the app on an iPhone 14 with the latest iOS version.\n2. Navigate to the checkout page.\n3. Attempt to apply a discount code using Safari.\n\nINVESTIGATION CHECKLIST:\n- [ ] Review Safari-specific JavaScript interactions and potential conflicts.\n- [ ] Investigate checkout API calls and server-side processing for any errors or timeouts.\n- [ ] Check for any iOS-specific security restrictions or limitations that may be causing the freeze.\n\nDATA GAPS FOR ENGINEERING:\n- More detailed logs or error messages from the iOS device to understand the cause of the freeze.\n- Information on the specific discount codes used and any variations that may affect the issue.\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: This issue has resulted in lost orders and has a high blast radius, impacting multiple users on a critical path of the application.""",
#    "F2_perf_api": """=== ENGINEERING INSIGHT ===\nTITLE: Increased API response times on Enterprise plan\nTICKET_TYPE: PERFORMANCE\nSEVERITY: P2-HIGH\nTECHNICAL SUMMARY: API response times have degraded from 200ms to 800ms, affecting production systems on the Enterprise plan. This change is likely due to an increase in latency, impacting the overall system performance. Further investigation is required to determine the root cause.\n\nAFFECTED COMPONENTS:\n- System: API\n- Platform: Enterprise plan\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. Check API response times for the past month to confirm the degradation.\n2. Verify if the issue is specific to the Enterprise plan or affects other plans as well.\n\nINVESTIGATION CHECKLIST:\n- Review API logs for any changes or anomalies in the past month\n- Check for any recent code deployments or changes to the API\n- Investigate resource utilization and CPU/Memory usage on the API servers\n\nDATA GAPS FOR ENGINEERING:\n- Detailed analysis of API request patterns and volumes\n- Information on recent infrastructure changes or upgrades\n- Any relevant metrics or logs from the past month\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The degradation in API response times has a significant impact on production systems, affecting the overall performance and potentially causing user experience issues. Given the Enterprise plan is affected, this issue requires immediate attention to ensure minimal disruption to critical systems.""",
#    "F3_usability_onboarding": """=== ENGINEERING INSIGHT ===\nTITLE: Confusing onboarding flow and broken getting started guide\nTICKET_TYPE: BUG\nSEVERITY: MEDIUM\nTECHNICAL SUMMARY: The onboarding flow is experiencing issues, causing difficulty in creating a first project. The getting started guide links to a 404 page, and YouTube tutorials are showing a different UI than the current application. This discrepancy is affecting users on the free plan, running Chrome on Windows.\n\nAFFECTED COMPONENTS:\n- System: Onboarding flow\n- Platform: Chrome\n- Environment: Windows\n\nREPRODUCTION STEPS:\n1. Signed up for the service yesterday\n2. Attempted to follow the getting started guide\n3. Encountered a 404 page\n4. Watched three YouTube tutorials that showed a different UI\n5. Still unable to create a first project\n\nINVESTIGATION CHECKLIST:\n- [ ] Verify the getting started guide URL is correct\n- [ ] Check for any recent changes to the onboarding flow or UI\n- [ ] Compare the current UI with the YouTube tutorials to identify discrepancies\n- [ ] Investigate why the YouTube tutorials are showing a different UI\n\nDATA GAPS FOR ENGINEERING:\n- More information about the exact steps taken during onboarding\n- Details about the exact YouTube tutorials watched (e.g., video IDs, timestamps)\n- Any error messages or logs related to the 404 page\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The onboarding flow is critical for new users, and a confusing experience can lead to user frustration and potential churn.""",
#    "F4_security": """=== ENGINEERING INSIGHT ===\nTITLE: Unauthorized login attempt from Romania\nTICKET_TYPE: SECURITY\nSEVERITY: HIGH\nTECHNICAL SUMMARY: An unauthorized login attempt was made to an account (lisa.t@company.com, #AC-41209) from an unknown device in Romania, despite the account having a changed password and enabled 2FA. The incident occurred at 3:47 AM EST. We need to determine if any data was accessed and perform a 30-day audit log review.\n\nAFFECTED COMPONENTS:\n- System: Authentication\n- Platform: Web\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. An unknown individual attempted to log in to the account lisa.t@company.com from IP address 203.0.113.42 in Romania at 3:47 AM EST.\n2. The login attempt was made despite the account having a changed password and enabled 2FA.\n\nINVESTIGATION CHECKLIST:\n- Check the authentication logs for any successful login attempts or data access\n- Review the 30-day audit log for the account #AC-41209\n- Investigate if any other accounts were accessed or affected by the same IP address\n\nDATA GAPS FOR ENGINEERING:\n- More information about the device used for the login attempt\n- Details about the login attempt, such as the user agent and browser type\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The unauthorized login attempt has high severity due to the potential data exposure and security breach. We need to act quickly to determine the extent of the incident and prevent further unauthorized access.""",
#    "F5_vague_angry": """== ENGINEERING INSIGHT ===\nTITLE: Unclear technical issue with app functionality\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: UNKNOWN\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. User reports the app is "garbage" and a "waste of money", but no specific technical details are provided.\n\nAFFECTED COMPONENTS:\n- System: Entire application\n- Platform: Not specified\n- Environment: Not specified\n\nREPRODUCTION STEPS:\n1. User reports the app is not functioning as expected, but no specific steps are provided.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Request a clear description of the issue and any error messages encountered\n\nDATA GAPS FOR ENGINEERING:\n- Specific technical details about the issue, including any error messages, stack traces, or system logs\n- Information about the user's environment, including platform and device\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: The issue is unclear and no specific impact on users or data integrity is reported, but further investigation is needed to determine the nature of the issue.""",
#    "F6_vague_minimal": """=== ENGINEERING INSIGHT ===\nTITLE: Poor user experience with checkout process\nTICKET_TYPE: BUG\nSEVERITY: HIGH\nTECHNICAL SUMMARY: The user experienced a poor checkout experience, specifically stating that the process was 'terrible'. The exact nature of the issue is unclear, but it occurred during the checkout process. The user did not provide specific details about the issue.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: Web\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. The user attempted to check out on the website.\n2. The user reported a poor experience, but did not provide further details.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Review recent checkout process changes\n\nDATA GAPS FOR ENGINEERING:\n- Specific error messages or error codes encountered by the user\n- Steps taken by the user leading up to the issue\n- Any relevant system logs or error messages\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The user reported a high-severity issue with the checkout process, which is a critical part of the user experience. Further investigation is needed to determine the root cause and prevent similar issues in the future.""",
#    "F7_vague_indifferent": """=== ENGINEERING INSIGHT ===\nTITLE: User reports issue with the app experience\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: Unknown - single report\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. Further details are required to determine the nature of the issue.\n\nAFFECTED COMPONENTS:\n- System: Unknown\n- Platform: Unknown\n- Environment: Unknown\n\nREPRODUCTION STEPS:\n1. User reported the issue with the app experience, but no specific steps were provided.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Gather information about the user's environment, platform, and system\n\nDATA GAPS FOR ENGINEERING:\n- Specific details about the issue\n- Technical information about the user's environment, platform, and system\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: Insufficient information is available to determine the priority of this issue. Further investigation is required to understand the scope and impact.""",
#    "F8_vague_demand": """=== ENGINEERING INSIGHT ===\nTITLE: iOS App Crashes When Applying Discount Codes\nTICKET_TYPE: BUG\nSEVERITY: HIGH\n\nTECHNICAL SUMMARY: The iOS app crashes when users attempt to apply discount codes on the checkout page. The crash occurs on the latest version of iOS and Safari. Further investigation is required to determine the root cause.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: iOS\n- Environment: Safari\n\nREPRODUCTION STEPS:\n1. Users report that they were able to select a product and proceed to the checkout page.\n2. When attempting to apply a discount code on the checkout page, the app crashes.\n\nINVESTIGATION CHECKLIST:\n- Check the app's crash logs for any relevant error messages or stack traces.\n- Review the code for the checkout page to ensure it is correctly handling discount code input.\n- Verify that the discount code API is functioning correctly.\n\nDATA GAPS FOR ENGINEERING:\n- Crash logs from the affected users\n- Additional details on the specific discount code being used when the crash occurred\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The crash is preventing users from completing the checkout process, which is a critical path for revenue generation. The high severity of the issue warrants immediate attention to minimize the impact on users and business operations.""",
#}

# sample outputs for testing the evaluation framework using F1-F8 test use-cases where output is collected using fine tuned models.
PIPELINE_OUTPUTS = {
     "F1_bug_checkout": """=== ENGINEERING INSIGHT === TITLE: Checkout freeze on iOS Safari when applying discount codes TICKET_TYPE: BUG SEVERITY: P2-HIGH

TE An iOS Safari user reports a 100% reproduction rate freeze when attempting to apply discount codes on the checkout page using an iPhone 14 with the latest iOS version. The freeze is blocking order completion, resulting in 3 lost orders in one week.

AFFECTED COMPONENTDateS:

System: Frontend (Checkout component)
Platform: iOS (Safari)
Environment: iPhone 14, latest iOS version
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps

Open the checkout page on iPhone 14 with Safari (latest iOS)
Add items to cart and proceed to checkout
Attempt to apply a discount code
Page freezes completely (user reports unable to interact with the page, cannot navigate away)
INVESTIGATION CHECKLIST:

[ ] Check Safari console logs for any JavaScript errors during the discount code application
[ ] Verify if the freeze occurs on other iOS devices/Safari versions
[ ] Check if the freeze happens with all discount codes or specific codes
[ ] Review recent changes to the discount code application logic
[ ] Check networkanship between the freeze and discount code API calls
DATA GAPS FOR ENGINEERING:

Exact discount code triggering the freeze (if it's specific to certain codes)
Whether the freeze is a full page hang or a JavaScript freeze
Whether other iOS devices/Safari versions are affected
Server-side logs for discount code API calls during the freeze
PRIORITY RECOMMENDATION: P2-HIGH PRIORITY JUST Turning P2-HIGH - user reports 100% reproduction on a production environment, 3 lost orders in one week, and the freeze is on the checkout page. This blocks revenue and has a high blast radius.
""",
    
     "F2_perf_api": """=== ENGINEERING INSIGHT === TITLE: API p95 latency regression from 200ms to 800ms affecting Enterprise plan TICKET_TYPE: BUG SEVERITY: P1-CRITICAL

TECHNICAL SUMMARY: API response times have degraded by 400% since last month, affecting all API requests for an Enterprise customer. p95 latency has increased from 200ms to 800ms. This regression has been reported across 3 separate support tickets, indicating it is a systemic issue and not account-specific.

AFFECTED COMPONENTS:

System: API Gateway
Platform: Unknown (API team would know infrastructure stack)
Environment: Production, Enterprise plan
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps:

Make API requests to the production API (specific endpoints not mentioned by user)
Observed p95 latency at 800ms (up from 200ms)
This regression started last month
3 separate support tickets confirm the regression
INVESTIGATION CHECKLIST:

[ ] Review API request volume and error rates for the past 60 days for signs of changes
[ ] Check for changes in instance count, scaling, or resource utilization on the API nodes
[ ] Review recent deploy history for API service for any changes, bug fixes, or config updates
[ ] Check for changes in database query performance or schema
[ ] Verify if the regression is consistent across all endpoints or specific to certain fleets
DATA GAPS FOR ENGINEERING:

Specific API endpoints affected
Request volumes and error rates pre-regression vs post-regression
API infrastructure stack (containerization, instance type, load balancer)
Recent deploy history for the API service
Database query performance changes
PRIORITY RECOMMENDATION: P1-CRITICAL PRIORITY JUSTIFICATION: 400% regression in p95 latency on a production API for an Enterprise customer is a showstopper. 3 confirmations across separate tickets indicate this is a systemic issue, not account-specific. API latency directly impacts revenue-generating capabilities for the customer.""",
    
     "F3_usability_onboarding": """=== ENGINEERING INSIGHT === TITLE: Incomplete onboarding - 404 on getting started guide, outdated YouTube tutorial UI matches TICKET_TYPE: BUG SEVERITY: P3-MEDIUM

TECHNICAL SUMMARY: The getting started guide link in the onboarding flow returns a 404 error. External YouTube tutorials show a UI that matches what the user sees, indicating the actual UI has regression or has changed since the video was created. This blocks a new Free plan user from creating their first project.

AFFECTED COMPONENTS:

System: Frontend (onboarding flow, getting breaking guide link)
Platform: Chrome on Windows
Environment: Free plan account, created yesterday
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps

Signed up for Free plan yesterday
Completed onboarding
Clicked through the getting started guide link
Served a 404 page
Watched 3 YouTube tutorials showing a different UI than current onboarding
INVESTIGATION CHECKLIST:

[ ] Check the current URL and status code of the getting started guide link
[ ] Verify the expected content exists and is up-to-date
[ ] Check if the link is redirecting correctly through CDNs or redirects
[ ] Check the UI diff between the current onboarding and the UI shown in the YouTube tutorials
[ ] Check if the regression is specific to Free plan or all plans
DATA GAPS FOR ENGINEERING:

Exact URL of the 404 page
HTTP status code of the 404
Onboarding flow steps that lead to the 404 (what specifically was clicked)
Account ID for server-side investigation
Exact YouTube video IDs for the tutorials
PRIORITY RECOMMENDATION: P3-MEDIUM PRIORITY JUSTIFICATION: Onboarding is the first user experience. A 404 on the getting started guide and outdated tutorial UI means a new user cannot complete onboarding. This blocks the user from activating their account and startingaccording to our CAC metrics.""",
    
     "F4_security": """=== ENGINEERING INSIGHT === TITLE: Unauthorized login from Romania on account AC-41209 with compromised credentials TICKET_TYPE: BUG SEVERITY: P2-HIGH

TECHNICAL SUMMARY: A login alert detected an unauthorized access attempt from IP 203.0.113.42 (Romania) at 3:47 AM EST on account AC-41209 (lisa.t@company.com). The user reports they changed their password and enabled 2FA, suggesting their credentials were compromised. We need to determine if account data was accessed and provide the user with a 30-day audit log and information on whether other accounts were affected.

AFFECTED COMPONENTS:

System: Authentication
Platform: Not platform-specific (cross-platform issue)
Environment: Production environment, account AC-41209
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps

User reports no login session started after changing password and enabling 2FA at an unknown previous time.
Login alert detected access from IP 203.0.113.42 (Romania) at 3:47 AM EST
INVESTIGATION CHECKLIST:

[ ] Check authentication logs for the account AC-41209 for all login attempts since the password change
[ ] Verify the new password hash and 2FA secrets were not compromised
[ ] Check for session fixation or token reuse
[ ] Run IP lookup and threat intelligence on 203.0mtime.113.42
[ ] Check for any other accounts with the same password hash compromised in the last 90 days
[ ] Review account activity for AC-41209 for the past 30 days
DATA GAPS FOR ENGINEERING:

Exact time of password compromise (before or after the user changed the password)
Whether data was accessed or what data was accessed
Session duration and actions performed during the unauthorized session
Whether the 2FA was bypassed or the authenticator token was captured
Whether the compromised credentials were used on other accounts
PRIORITY RECOMMENDATION: P2-HIGH PRIORITY JUSTIFICATION: Data breach, potential PII exposure. User took security precautions after compromise detected, indicating the attacker likely had access for some time. Need to confirm data access and scope other account compromise.""",
    
     "F5_vague_angry": """=== ENGINEERING INSIGHT === TITLE: Unactionable complaint - no technical details TICKET_TYPE: INVESTIGATION_NEEDED SEVERITY: P ilk-LOW

TECHNICAL SUMMARY: Insufficient data for engineering action. No specific feature, system, or behavior is described. AFFECTED COMPONENTS:

System: UNKNOWN
Platform: UNKNOWN
Environment: UNKNOWN
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps

User says the app is garbage and a waste of money but provides no specifics.
INVESTIGATION CHECKLIST:

[ ] Reach out to user for technical details (what, when, how)
DATA GAPS FOR ENGINEERING: No system, platform, environment, error, feature, or behavior is mentioned. No repro steps at all.

PRIORITY RECOMMENDATION: P4-LOW PRIORITY JUSTIFICATION: No actionable information for engineering. Cannot investigate what is not described.""",
    
     "F6_vague_minimal": """Cannot write what I do not know. What is the user feedback? Provide the classified user feedback and I will write a structured engineering insight.""",
    
     "F7_vague_indifferent": """INSUFFICIENT DATA for engineering. Cannot write structured insight without user behavior. What would you like to add to provide engineering context? What were you trying to do, what happened, on what platform, and under what conditions?""",
    
     "F8_vague_demand": """=== ENGINEERING INSIGHT === TITLE: Unavailable - no feedback provided TICKET_TYPE: UNCLEAR SEVERITY: UNKNOWN

TECHNICAL SUMMARY: Insufficient data for engineering. No issue description, system, platform, or environment provided.

AFFECTED COMPONENTS:

System: N/A
Platform: N/A
Environment: N/A
REPRODUCTION STEPS: INSUFFICIENT DATA - steps below are what we know, not confirmed reproduction steps

No steps available
INVESTIGATION CHECKLIST:

[ ] Reach out to user for technical details
DATA GAPS FOR ENGINEERING: Everything. No issue, system, platform, or steps provided.

PRIORITY RECOMMENDATION: P4-LOW PRIORITY JUSTIFICATION: Cannot investigate without any data.""",
    
}

# ============================================================
# METRICS
# ============================================================
engineering_groundedness = GEval(
    name="Engineering Groundedness",
    criteria=(
        "The input is detailed user feedback with specific technical details. "
        "Does the engineering insight ONLY use information from this feedback? "
        "1) REPRODUCTION STEPS must only describe actions the user actually mentioned. "
        "2) No invented internal system names or architecture details the user did not mention "
        "(e.g., do not invent 'API Gateway' or 'GraphQL' if the user just said 'API'). "
        "3) No fabricated user counts, percentages, or revenue impact. "
        "4) DATA GAPS section should list what is NOT known. "
        "Grounded = PASS. Invented technical specifics = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

engineering_usefulness = GEval(
    name="Engineering Usefulness",
    criteria=(
        "Would an engineer find this insight useful for starting an investigation? "
        "Does it identify the affected system based on what the user described? "
        "Does it provide a reasonable investigation checklist? "
        "Does it clearly state data gaps? "
        "Useful engineering ticket = PASS. Vague or unhelpful = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

vague_handling = GEval(
    name="Vague Feedback Handling",
    criteria=(
        "The input is vague or lacks technical details. A CORRECT response should: "
        "1) Identify that engineering cannot act without more information. "
        "2) Set affected components to UNKNOWN. "
        "3) NOT invent systems, platforms, error types, or root causes. "
        "4) NOT fabricate reproduction steps. "
        "5) Recommend reaching out to the user for clarification. "
        "6) Set priority to LOW since nothing is actionable. "
        "The checklist CAN include multiple follow-up items about GATHERING info "
        "(ask for platform, request screenshots) as long as no specific problem is invented. "
        "Correctly flagging as unactionable = PASS. Inventing details = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")

def run_evaluation():
    print("\n" + "=" * 60)
    print("EVALUATING DETAILED FEEDBACK (4 test cases)")
    print("=" * 60)
    detailed = [LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[f])
                for f, d in TEST_FEEDBACK.items()
                if d["label"] == "detailed" and "PASTE" not in PIPELINE_OUTPUTS.get(f, "PASTE")]
    if detailed:
        evaluate(test_cases=detailed, metrics=[engineering_groundedness, engineering_usefulness, relevancy])

    print("\n" + "=" * 60)
    print("EVALUATING VAGUE FEEDBACK (4 test cases)")
    print("=" * 60)
    vague = [LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[f])
             for f, d in TEST_FEEDBACK.items()
             if d["label"] == "vague" and "PASTE" not in PIPELINE_OUTPUTS.get(f, "PASTE")]
    if vague:
        evaluate(test_cases=vague, metrics=[vague_handling, relevancy])

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nWhat to look for:")
    print("  Detailed: Groundedness + Usefulness should PASS")
    print("  Vague:    Vague Handling should PASS")
    print("  All:      Relevancy should PASS")
    print("\nSave a screenshot for before/after comparison.")

if __name__ == "__main__":
    print("=" * 60)
    print("FEEDBACK TO ENGINEERING INSIGHTS - DeepEval Evaluation")
    print("=" * 60)
    run_evaluation()