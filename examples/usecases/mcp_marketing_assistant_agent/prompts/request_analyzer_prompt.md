# Request Analysis Prompt

Analyze this marketing content request for **{company_name}** to determine if clarification is needed.

## Company Context
- **Name**: {company_name}
- **Industry**: {industry}
- **Messaging Pillars**: {messaging_pillars}
- **Target Audience**: {target_audience}
- **Brand Personality**: {brand_personality}

## Content Request Details
- **Platform**: {platform}
- **Request**: "{request}"
- **Context Available**: {context_length} characters

## Analysis Framework

### Primary Evaluation Criteria

**Analyze the request for completeness and clarity:**

1. **Request Specificity**
   - Is the content goal clear? (awareness, conversion, engagement, etc.)
   - Is there enough detail to create quality content?
   - Are key requirements specified?

2. **Campaign Context**
   - Is the purpose/objective obvious?
   - Is this tied to a specific campaign or event?
   - Are there timing considerations?

3. **Target Audience Clarity**
   - Is it clear who this targets within our audience?
   - Is the audience's knowledge level apparent?
   - Are there specific demographic considerations?

4. **Content Requirements**
   - Is there a specific message or announcement?
   - Are key products/services/features mentioned?
   - Is a call-to-action specified or obvious?

5. **Platform Optimization**
   - Does the request consider {platform} best practices?
   - Is the appropriate tone/style clear?
   - Are platform-specific requirements addressed?

### Clarification Decision Matrix

**NEEDS CLARIFICATION** when request:
- Is very short (under 6 words) and vague
- Has unclear or multiple possible goals
- Lacks specific direction for content focus
- Missing obvious call-to-action for conversion content
- References unclear events, products, or campaigns
- Has ambiguous audience within our target market
- Requests content type that doesn't match platform norms

**NO CLARIFICATION NEEDED** when request:
- Provides clear, specific direction
- Includes obvious goal and context
- References specific products/services/events
- Has sufficient detail for quality content creation
- Includes URL with relevant detailed context
- Goal is obvious from context and platform
- Standard content type for the platform

**BORDERLINE CASES** (Use judgment):
- Medium-length requests with some ambiguity
- Clear topic but unclear angle or goal
- Obvious audience but unclear message priority
- Platform-appropriate but generic requests

### Smart Question Generation

**Only ask questions that will meaningfully improve content quality.**

#### High-Impact Questions (Ask these first)
- **Campaign Goal**: "What's the main objective?" 
  - When: Goal isn't obvious from request
  - Options: awareness, lead generation, event promotion, thought leadership, product announcement

- **Specific Focus**: "What should we highlight about [topic]?"
  - When: Topic is broad or could have multiple angles
  - Options: benefits, features, use cases, success stories

- **Call-to-Action**: "What should readers do next?"
  - When: Next step isn't clear for conversion content
  - Options: visit website, sign up, download, attend event, contact sales

#### Medium-Impact Questions (Ask if space allows)
- **Audience Segment**: "Who specifically within our {target_audience}?"
  - When: Multiple audience segments could apply
  - Options: specific roles, experience levels, company sizes

- **Tone Preference**: "What tone works best for this?"
  - When: Multiple tones could work for the platform
  - Options: professional, casual, technical, friendly

- **Campaign Context**: "Is this connected to a broader campaign?"
  - When: Content might be part of larger initiative
  - Options: product launch, event promotion, thought leadership series

#### Low-Impact Questions (Usually skip)
- Basic platform requirements (we know these)
- Company background information (we have this)
- Generic messaging (covered in brand guidelines)

### Context Utilization

**Before requesting clarification, check if context provides:**
- Company documentation that clarifies intent
- URL content that explains the topic
- Content samples that suggest appropriate style
- Previous similar requests that provide patterns

**If context is rich and detailed, reduce clarification needs.**

## Response Format

Respond in JSON format:

```json
{{
  "needs_clarification": true/false,
  "confidence": "high/medium/low",
  "questions": [
    {{
      "key": "campaign_goal",
      "question": "What's the main goal of this content?",
      "type": "required/optional",
      "suggestions": ["awareness", "conversion", "engagement"],
      "impact": "high/medium/low"
    }}
  ],
  "reason": "Brief explanation of why clarification is/isn't needed",
  "auto_assumptions": {{
    "likely_goal": "best guess based on request and platform",
    "probable_audience": "most likely audience segment",
    "suggested_cta": "recommended call-to-action"
  }},
  "context_assessment": {{
    "sufficient_detail": true/false,
    "clear_objective": true/false,
    "platform_appropriate": true/false
  }}
}}
```

## Quality Guidelines for Analysis

### Be Efficient
- Maximum 3 questions total
- Prefer 1-2 high-impact questions
- Only ask what's truly needed for quality content

### Be Smart
- Use context to fill information gaps
- Make reasonable assumptions based on platform norms
- Consider {company_name}'s typical content patterns

### Be Helpful
- Provide specific, actionable questions
- Offer multiple choice answers when possible
- Include reasoning for why information is needed

### Be Respectful
- Don't ask for information that's already clear
- Respect the user's time and expertise
- Focus on content improvement, not perfect information

## Platform-Specific Considerations

### {platform} Typical Patterns
- **LinkedIn**: Professional updates, thought leadership, company news
- **Twitter**: Quick updates, engaging conversations, trending topics
- **Email**: Direct communication, newsletters, announcements
- **Instagram**: Visual storytelling, behind-scenes, lifestyle content

**Adjust clarification needs based on platform norms and {company_name}'s typical content strategy.**

---
**Analysis Date**: {current_date}
**Target**: {platform} content for {company_name}
**Audience**: {target_audience}
**Context**: {context_length} characters available