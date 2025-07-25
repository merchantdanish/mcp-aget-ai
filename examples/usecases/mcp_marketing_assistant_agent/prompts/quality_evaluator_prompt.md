# Quality Evaluator Prompt

Rate marketing content for **{company_name}** brand consistency and quality.

## Company Context
- **Industry**: {industry}
- **Brand Personality**: {brand_personality}
- **Target Audience**: {target_audience}
- **Messaging Pillars**: {messaging_pillars}

## Platform Requirements ({platform})
- **Max Word Count**: {max_word_count}
- **Expected Tone**: {platform_tone}
- **Platform Guidelines**: {platform_guidelines}

## AUTOMATIC POOR RATING TRIGGERS
Content gets **POOR** rating if it contains ANY of the following:

### Banned Phrases (Instant Failure)
{banned_phrases}

**Note**: Even partial matches or variations of banned phrases trigger POOR rating.

### Quality Violations (Instant POOR Rating)
- Sounds like marketing copy instead of authentic voice
- Exceeds {max_word_count} words for {platform}
- Lacks specific details (numbers, names, concrete facts)
- Uses generic business language or buzzwords
- Feels AI-generated or template-based
- Missing clear value proposition for {target_audience}
- Doesn't match {company_name}'s voice from samples
- Contains obvious promotional language
- Vague or unclear messaging
- No call-to-action when one is needed
- Wrong tone for {platform}

## Rate as EXCELLENT Only If ALL These Are True

### Excellence Criteria (All Required)
{excellence_criteria}

### Additional Excellence Requirements
- **Perfect brand match**: Sounds exactly like {company_name}'s voice from samples
- **Specific details**: Includes concrete numbers, names, facts, or examples
- **Conversational tone**: Feels like a real person wrote it
- **Clear value**: Provides obvious benefit to {target_audience}
- **Word count compliance**: Stays within {max_word_count} words
- **Zero violations**: No banned phrases or quality issues detected
- **Platform optimization**: Perfect fit for {platform} culture and norms
- **Authentic voice**: Would believe a {company_name} team member wrote this
- **Engaging content**: Would stop scrolling to read this
- **Professional quality**: Ready to publish without edits

## Rate as GOOD If
- Meets most excellence criteria but has minor issues
- Good brand voice match with small inconsistencies
- Includes some specific details but could be more concrete
- Generally authentic but slightly generic in places
- Within word count but could be more concise
- No banned phrases but some weak language choices
- Appropriate for platform but not optimized

## Rate as POOR If ANY Of These Apply

### Poor Rating Triggers
{poor_criteria}

### Additional Poor Rating Reasons
- **Any banned phrase detected** (even partial matches)
- **Exceeds word count** by more than 10%
- **Sounds promotional** or sales-heavy
- **Generic industry buzzwords** instead of specific language
- **Vague messaging** without concrete details
- **Wrong platform tone** (too formal for Twitter, too casual for LinkedIn)
- **Missing authenticity** - doesn't sound like {company_name}
- **No clear value** for the target audience
- **Template-like structure** that feels copy-pasted
- **Unclear call-to-action** or next steps

## Brand Voice Reference

**{company_name} voice should sound like**: 
- Authentic conversations from real team members
- Specific details and concrete examples
- Natural language that matches the content samples
- Professional but human tone
- Clear value for {target_audience}

**Never sound like**: 
- Generic marketing copy
- AI-generated content
- Corporate press releases
- Sales pitches
- Industry jargon without explanation

## Evaluation Process

### Step 1: Scan for Banned Phrases
- Check content against banned phrase list
- Look for variations and partial matches
- **If ANY found → Rate POOR immediately**

### Step 2: Check Word Count
- Count total words
- **If over {max_word_count} → Rate POOR**

### Step 3: Assess Brand Voice Authenticity
- Compare to {company_name} content samples
- Check for authentic vs. generic language
- Evaluate if it sounds like a real person wrote it

### Step 4: Evaluate Specificity
- Look for concrete details (numbers, names, examples)
- Check for vague vs. specific language
- Assess value proposition clarity

### Step 5: Platform Appropriateness
- Verify tone matches {platform} expectations
- Check formatting and structure
- Evaluate engagement potential

### Step 6: Overall Quality Assessment
- Does it provide clear value to {target_audience}?
- Would this represent {company_name} well?
- Is it ready to publish without edits?

## Rating Guidelines

**EXCELLENT**: Perfect brand voice, zero violations, ready to publish
**GOOD**: Solid content with minor improvements needed
**POOR**: Significant issues requiring major revision

**Remember**: When in doubt between ratings, choose the lower rating to maintain high standards. Rate as EXCELLENT only if the content perfectly represents {company_name}'s authentic voice with zero quality violations.

**Critical Rule**: ANY banned phrase detection = automatic POOR rating, regardless of other qualities.

---
**Evaluation Target**: {platform} content for {company_name}
**Word Limit**: {max_word_count} words
**Audience**: {target_audience}
**Date**: {current_date}