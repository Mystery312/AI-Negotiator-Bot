# Comprehensive Business Negotiation Demo Scenario

## Scenario Overview

**Context**: Two department heads are negotiating resource allocation for a critical Q2 project launch. They must agree on how to divide limited company resources.

**Parties**:
- **Alex** (Product Development) - Needs resources to ship new features
- **Jordan** (Operations) - Needs resources to maintain infrastructure stability

**Resources to Negotiate**:
1. **Senior Engineers** (Total: 5 available)
   - Highly skilled developers who can work on either product features or infrastructure
   - Most valuable resource for both parties

2. **Budget ($K)** (Total: 200K available)
   - Discretionary spending for contractors, tools, or cloud resources
   - Flexible but limited

3. **Timeline (weeks)** (Total: 12 weeks until deadline)
   - How many weeks each team gets priority access to shared resources
   - Time is fixed but can be allocated differently

**Goal**: Reach a mutually beneficial agreement that maximizes value for both departments

---

## Configuration Setup

Before starting the negotiation, configure the chatbot:

```
⚙️ Item Configuration:
├── Item 1 Name: Senior Engineers
├── Item 1 Quantity: 5
├── Item 2 Name: Budget ($K)
├── Item 2 Quantity: 200
├── Item 3 Name: Timeline (weeks)
└── Item 3 Quantity: 12
```

---

## 15-Turn Negotiation Script

### **Turn 1: Alex (Opening - Anchoring High)**
**Message**:
> "Hi Jordan, thanks for meeting. I've reviewed our Q2 priorities and I believe Product Development should get 4 Senior Engineers, $150K budget, and 8 weeks of priority timeline. We have critical customer features that can't slip."

**AI Coach Analysis** (for Jordan):
- **Move**: Strong opening anchor with aggressive position
- **Power Dynamics**: Alex is showing confidence (C)
- **Advice**: "Alex opened with a very aggressive anchor (80% engineers, 75% budget, 67% timeline). Don't accept this immediately. Probe their underlying needs - which features are truly critical? Counter with your own priorities."
- **Item Priority Detection**:
  - Alex highly values: Senior Engineers (mentioned first, quantity 4)
  - Budget also important (specific amount $150K)
  - Timeline mentioned but may be flexible

---

### **Turn 2: Jordan (Counter - Establishing Value)**
**Message**:
> "I understand the customer features are important, but infrastructure stability directly impacts ALL customers. I need at least 3 Senior Engineers to prevent outages, $120K for our database migration, and 7 weeks to complete the work properly."

**AI Coach Analysis** (for Alex):
- **Move**: Counter-anchor with justification
- **Power Dynamics**: Jordan asserting equal priority (C)
- **Advice**: "Jordan countered with strong justification (customer impact). Their proposal: 3 engineers (60%), $120K (60%), 7 weeks (58%). Notice they didn't challenge your timeline claim. Possible trade: You might get more timeline if you give up some engineers."
- **Pareto Suggestion**: "Consider: 3 Senior Engineers, $110K Budget, 8 Timeline weeks for you. This gives Jordan engineers they need while you keep timeline priority."

---

### **Turn 3: Alex (Probing - Information Gathering)**
**Message**:
> "I hear you on the infrastructure concern. Help me understand - which specific systems are at risk? And could the database migration be done with contractors instead of our Senior Engineers?"

**AI Coach Analysis** (for Jordan):
- **Move**: Information request (probing)
- **Power Dynamics**: Collaborative (P - Procedural)
- **Advice**: "Alex is probing your needs - this is a positive sign they're open to trade-offs. Be specific about technical constraints. If you can use contractors for some work, you might trade budget for more engineers."
- **Strategic Tip**: "Transparency here builds trust. Share real constraints but keep negotiating leverage."

---

### **Turn 4: Jordan (Information Sharing - Building Trust)**
**Message**:
> "Fair question. Our payment processing system needs a complete rewrite - that requires Senior Engineers, not contractors. However, the monitoring tools upgrade could be done with contractors for about $60K. That's why I need the $120K budget."

**AI Coach Analysis** (for Alex):
- **Move**: Detailed information sharing
- **Power Dynamics**: Collaborative (P)
- **Advice**: "Jordan revealed important information: Payment system is non-negotiable (needs senior engineers), but monitoring is flexible (contractors ok). You could propose: They get 3 engineers for payments, you get 2 engineers for features, and you help them secure contractor budget."
- **Pareto Suggestion**: "Proposed split: You get 2 Senior Engineers, $80K Budget, 8 Timeline weeks. Jordan gets 3 Senior Engineers, $120K Budget, 4 Timeline weeks. Total value increased by separating engineer-intensive vs budget-intensive work."

---

### **Turn 5: Alex (Proposal - Creating Value)**
**Message**:
> "Okay, I think I see a path forward. What if we split the engineers 2 for me, 3 for you since payments are critical. But I'd need $100K for hiring contractors to handle our less complex features, and 7 weeks of priority access."

**AI Coach Analysis** (for Jordan):
- **Move**: Proposal with trade-offs
- **Power Dynamics**: Collaborative (P)
- **Advice**: "Alex is proposing: 2 engineers (40%), $100K (50%), 7 weeks (58%). This is movement toward you on engineers! But they're asking for half the budget and more timeline. Counter: You could accept 2 engineers if you get more budget ($120K) and equal timeline split (6 weeks each)."
- **Negotiation Insight**: "Both parties are now problem-solving together - this is optimal negotiation state."

---

### **Turn 6: Jordan (Counter-Proposal - Fine-Tuning)**
**Message**:
> "I can work with 2/3 engineer split. But $100K leaves me only $100K, and I need $120K minimum for the migration. How about: You get 2 engineers, $90K, and 7 weeks. I get 3 engineers, $110K, and 5 weeks?"

**AI Coach Analysis** (for Alex):
- **Move**: Counter-proposal with small adjustments
- **Power Dynamics**: Balanced (P)
- **Advice**: "Jordan accepted your engineer split! They're asking for $10K more budget ($110K vs $100K) and giving you extra timeline (7 vs 5 weeks). This is close to a deal. Consider: Timeline may be worth more to you than $10K. Can you find budget elsewhere?"
- **Pareto Check**: "Current proposal is near-optimal. Jordan values engineers most, you value timeline most. Budget is the remaining sticking point."

---

### **Turn 7: Alex (Probing - Finding Creative Solutions)**
**Message**:
> "The timeline split works for me - 7 weeks is enough. On budget, I'm tight. Could we structure it differently? What if I get $85K now, but you get first claim on any remaining budget at end of Q2?"

**AI Coach Analysis** (for Jordan):
- **Move**: Creative proposal (future consideration)
- **Power Dynamics**: Collaborative (P)
- **Advice**: "Alex is offering a contingent deal: $85K now + potential future budget vs guaranteed $90K. Risk: 'Remaining budget' might be $0. Counter: Accept if they guarantee a minimum ($10K minimum remaining) or ask for $95K guaranteed instead."
- **Risk Assessment**: "Contingent deals create uncertainty. Value guaranteed resources higher."

---

### **Turn 8: Jordan (Reality Check - Establishing Constraints)**
**Message**:
> "I appreciate the creativity, but I need budget certainty for vendor contracts. I can't commit to contractors without knowing the full amount. What if we meet in the middle at $92.5K each? That gives you $107.5K total and me $92.5K."

**AI Coach Analysis** (for Alex):
- **Move**: Constraint statement + compromise proposal
- **Power Dynamics**: Assertive but fair (C)
- **Advice**: "Jordan has a legitimate constraint (vendor contracts need certainty). They're proposing nearly equal budget split: $107.5K for you (54%), $92.5K for them (46%). You're already getting 7/12 weeks (58%) and 2/5 engineers (40%). This is actually quite balanced overall."
- **Pareto Analysis**: "Resource allocation: Engineers favor Jordan (60/40), Budget favors you (54/46), Timeline favors you (58/42). This is a fair multi-dimensional deal."

---

### **Turn 9: Alex (Acceptance with Caveat)**
**Message**:
> "Alright, $107.5K works for me if we can formalize the priority timeline clearly. I need confirmation that weeks 1-7, my team gets first access to the Senior Engineers for critical questions, even if they're assigned to you."

**AI Coach Analysis** (for Jordan):
- **Move**: Conditional acceptance
- **Power Dynamics**: Clarification seeking (P)
- **Advice**: "Alex is essentially accepting but wants to clarify 'priority timeline' meaning. They want ad-hoc access to engineers during your weeks. Important: Define boundaries. Counter: 'Priority timeline' means your team leads planning, but you'll provide 4 hours/week for cross-team support."
- **Implementation Tip**: "Vague agreements lead to future conflicts. Define operational details now."

---

### **Turn 10: Jordan (Clarification - Defining Terms)**
**Message**:
> "I'm good with the numbers, but let's be specific about 'priority access.' During my 5 weeks, my team sets the engineering roadmap and sprint planning. I can commit to 4 hours per week of ad-hoc support for urgent product questions. Does that work?"

**AI Coach Analysis** (for Alex):
- **Move**: Clarification with specific proposal
- **Power Dynamics**: Procedural (P)
- **Advice**: "Jordan is defining operational terms clearly: They control roadmap during their weeks, but provide 4 hours/week support. This is reasonable. Consider: Is 4 hours enough for your urgent needs? If yes, accept. If no, ask for 6-8 hours/week or define 'urgent' criteria."
- **Deal Structure**: "You're 90% to agreement. Don't derail over minor implementation details."

---

### **Turn 11: Alex (Refinement - Final Details)**
**Message**:
> "4 hours works, but let's define 'urgent': production bugs, critical customer escalations, or revenue-impacting issues. For those, can we get same-day response within the 4-hour window?"

**AI Coach Analysis** (for Jordan):
- **Move**: Operational detail specification
- **Power Dynamics**: Procedural (P)
- **Advice**: "Alex is defining 'urgent' criteria - this is good practice. Their list is reasonable (prod bugs, escalations, revenue). Same-day response is doable within 4 hours. Accept this to close the deal, or counter with 'next-business-day' if 4 hours is too tight."
- **Closing Signal**: "You're negotiating implementation, not core terms. Deal is nearly done."

---

### **Turn 12: Jordan (Agreement - Locking In)**
**Message**:
> "Same-day response for production issues is totally reasonable. So to confirm our agreement:
> - **Engineers**: You get 2, I get 3
> - **Budget**: You get $107.5K, I get $92.5K
> - **Timeline**: You get weeks 1-7 (priority), I get weeks 8-12 (priority)
> - **Support**: 4 hours/week ad-hoc for urgent production issues
>
> Is that accurate?"

**AI Coach Analysis** (for Alex):
- **Move**: Deal summary and confirmation
- **Power Dynamics**: Procedural (P)
- **Advice**: "Jordan is summarizing the full agreement for confirmation. Review carefully:
  - Engineers: 2/5 (40%) ✓
  - Budget: $107.5K/200K (54%) ✓
  - Timeline: 7/12 weeks (58%) ✓
  - Support: Defined and reasonable ✓

If this matches your understanding, confirm and close. If anything is wrong, correct now."
- **Pareto Outcome**: "This deal is on the Pareto frontier - improving one party's outcome would harm the other. Well negotiated!"

---

### **Turn 13: Alex (Confirmation - Deal Closure)**
**Message**:
> "Yes, that's exactly right. I'm good with this agreement. Should we document this in an email and cc our VP to make it official?"

**AI Coach Analysis** (for Jordan):
- **Move**: Confirmation + procedural next step
- **Power Dynamics**: Procedural (P)
- **Advice**: "Deal is closed! Alex is suggesting documentation (smart - prevents future disputes). Agree and propose adding:
  1. Start date (when does week 1 begin?)
  2. Monthly check-ins to adjust if priorities shift
  3. Escalation path if issues arise

Then formally close."

---

### **Turn 14: Jordan (Formalization - Documentation)**
**Message**:
> "Perfect! Yes, let's email the VP. I'll draft it and include:
> - Start date: Next Monday (April 3rd)
> - Monthly check-ins on the 1st to review progress
> - Escalation to VP if we can't resolve urgent issues within 24 hours
>
> I'll send the draft by EOD today. Thanks for working through this collaboratively, Alex."

**AI Coach Analysis** (for Alex):
- **Move**: Procedural details + relationship building
- **Power Dynamics**: Collaborative (P)
- **Advice**: "Jordan added good operational details. Review the draft email carefully before approving. Also note: They're building long-term relationship ('working collaboratively') - reciprocate to maintain positive working relationship for future negotiations."

---

### **Turn 15: Alex (Closure - Relationship Building)**
**Message**:
> "Sounds great, Jordan. I really appreciate your flexibility on the engineering split - I know payments are critical. Let's sync next week after the kick-off to make sure both teams are aligned. Looking forward to a successful Q2!"

**AI Coach Analysis** (for Jordan):
- **Move**: Relationship building + future coordination
- **Power Dynamics**: Collaborative (P)
- **Advice**: "Negotiation successfully closed! Alex is:
  1. Acknowledging your concession (good faith)
  2. Proposing follow-up (ensures execution)
  3. Expressing optimism (relationship building)

Respond positively to close on a high note."
- **Final Outcome Summary**:
  - **Alex**: 2 Senior Engineers, $107.5K, 7 weeks priority (40%/54%/58%)
  - **Jordan**: 3 Senior Engineers, $92.5K, 5 weeks priority (60%/46%/42%)
  - **Result**: Balanced multi-dimensional agreement, both parties satisfied, relationship preserved

---

## Key Coaching Insights Demonstrated

### **1. Move Detection**
- Anchoring (Turn 1, 2)
- Information gathering (Turn 3)
- Information sharing (Turn 4)
- Proposal/Counter-proposal (Turn 5-8)
- Clarification (Turn 9-11)
- Confirmation (Turn 12-13)
- Closure (Turn 14-15)

### **2. Power Dynamics**
- **Confidence (C)**: Turn 1, 2 - Strong opening positions
- **Procedural (P)**: Turn 3-15 - Collaborative problem-solving
- No **Defensive (D)** moves - indicates healthy negotiation

### **3. Pareto Optimization**
- Coach suggested win-win splits at multiple points
- Final deal is Pareto-optimal: Can't improve one party without harming other
- Multi-dimensional trade-offs (engineers vs budget vs timeline)

### **4. Item Priority Learning**
The AI correctly identified:
- **Alex priorities**: Timeline > Budget > Engineers (based on messaging patterns)
- **Jordan priorities**: Engineers > Budget > Timeline (based on "critical" language)
- This drove the optimal split recommendation

### **5. RAG (Retrieval-Augmented Generation)**
- Coach could reference similar negotiations from DOND dataset
- Generic negotiation tactics applied appropriately to business context

---

## How to Run This Demo

### **Step 1: Configure Items**
```
1. Open http://localhost:7860
2. Click "⚙️ Item Configuration"
3. Enter:
   - Item 1: "Senior Engineers" (quantity: 5)
   - Item 2: "Budget ($K)" (quantity: 200)
   - Item 3: "Timeline (weeks)" (quantity: 12)
4. Click "✓ Set Item Configuration"
```

### **Step 2: Set Up Conversation**
```
1. Select role: "Agent A" (Alex)
2. Enter Turn 1 message
3. Click "Send"
4. Review AI coach advice
```

### **Step 3: Continue Alternating**
```
1. Switch role to "Agent B" (Jordan)
2. Enter Turn 2 message
3. Review coach advice for Jordan
4. Switch back to "Agent A"
5. Continue through all 15 turns
```

### **Step 4: Analyze Results**
```
1. Check conversation history
2. Review Pareto suggestions
3. See how AI advice evolved with context
4. Verify custom item names appear throughout
```

---

## Expected AI Behavior

### **Early Turns (1-4)**
- Minimal advice ("Need more context")
- Once both parties speak: Move detection begins
- Item priority inference starts

### **Middle Turns (5-10)**
- Rich tactical advice
- Pareto suggestions appear
- Trade-off analysis
- Power dynamic tracking

### **Late Turns (11-15)**
- Deal closure guidance
- Implementation details
- Relationship building tips
- Final outcome summary

---

## Success Metrics

✅ **Custom Item Names**: "Senior Engineers" appears in all advice (not "item0")
✅ **Preference Learning**: AI infers Alex values timeline, Jordan values engineers
✅ **Pareto Suggestions**: AI proposes 2/3 engineer split around Turn 5-6
✅ **Move Detection**: Labels anchoring, proposals, clarifications correctly
✅ **Power Dynamics**: Tracks C → P transition (competitive to collaborative)
✅ **Context Building**: Advice becomes more specific with each turn

---

## Variations to Try

### **Scenario A: Aggressive Negotiation**
- Alex demands 4 engineers, refuses to budge
- Tests AI's advice on defensive tactics

### **Scenario B: Information Asymmetry**
- Alex knows budget will increase next quarter
- Tests AI's detection of hidden information

### **Scenario C: Impasse**
- Both parties stick to incompatible positions
- Tests AI's suggestion of creative solutions

### **Scenario D: Different Resources**
- Change to: "Cloud Credits", "API Rate Limits", "Storage (TB)"
- Validates system works with any resource types

---

## Technical Notes

**Item Name Matching**:
- AI detects "engineers" → maps to "Senior Engineers" ✓
- AI detects "budget" → maps to "Budget ($K)" ✓
- AI detects "timeline" or "weeks" → maps to "Timeline (weeks)" ✓
- Handles plurals, partial matches, multi-word names

**Quantity Handling**:
- "I need 2 Senior Engineers" → Extracts quantity: 2
- "$100K budget" → Extracts quantity: 100 (from 200 total)
- "7 weeks" → Extracts quantity: 7 (from 12 total)

**Context Window**:
- AI remembers last 5 turns
- Long negotiations benefit from Neo4j graph storage
- Can reference earlier discussion points

---

## Troubleshooting

**Issue**: AI says "Need more conversation context"
**Fix**: Ensure both agents have spoken at least once

**Issue**: Custom names not appearing
**Fix**: Verify Item Configuration was set before first message

**Issue**: Pareto suggestions seem wrong
**Fix**: Check that quantities are set correctly (engineers=5, not 500)

**Issue**: Coach advice is generic
**Fix**: Add more specific details in messages (numbers, reasons, priorities)

---

## Next Steps

After completing this demo:
1. Try the DOND Visualizer with custom item names
2. Test autoplay mode with these resources
3. Export conversation to Neo4j and analyze graph
4. Compare AI advice quality vs generic chatbots
5. Build your own negotiation scenarios!
