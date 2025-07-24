# Context Organizer Prompt

Organize and prioritize context for creating **{company_name}** marketing content.

## Request Analysis
- **Platform**: {platform}
- **Request**: "{request}"
- **Company**: {company_name}
- **Target Audience**: {target_audience}

## Available Context Sources
- **Company Docs**: {company_docs_length} characters
- **Content Samples**: {content_samples_length} characters  
- **URL Content**: {url_content_length} characters
- **Total Context**: Available for processing

## Context Organization Task

Your job is to intelligently organize and prioritize the most relevant context to help create exceptional marketing content for this specific request.

### Prioritization Framework

#### **HIGHEST PRIORITY** (Include first and preserve exactly)
1. **Platform-specific content samples** that match {platform}
2. **Brand voice examples** that demonstrate authentic {company_name} tone
3. **Messaging related to the request topic** (products, services, announcements)
4. **URL content directly relevant** to the request subject
5. **Specific style guidelines** for {platform}

#### **HIGH PRIORITY** (Include with smart summarization)
1. **Core brand messaging pillars** and value propositions
2. **Target audience insights** relevant to {target_audience}
3. **Product/service information** mentioned in or related to request
4. **Company positioning** and competitive advantages
5. **Recent announcements** or news relevant to request

#### **MEDIUM PRIORITY** (Include key points only)
1. **General company background** and mission
2. **Industry context** and market positioning
3. **Additional content samples** from other platforms
4. **Supporting documentation** that provides helpful context

#### **LOW PRIORITY** (Include only if essential and space allows)
1. **Historical content** that doesn't match current voice
2. **Technical documentation** not relevant to marketing
3. **Internal processes** or operational details
4. **Regulatory or legal content** unless specifically needed

### Organization Principles

#### **Preserve Authenticity**
- Keep content samples **exactly as written** - don't paraphrase or edit
- Maintain original voice examples word-for-word
- Preserve specific language patterns and style elements

#### **Maximize Relevance**
- Prioritize information directly related to the request
- Focus on content that will improve the final output quality
- Include context that demonstrates {company_name}'s authentic voice

#### **Optimize for Content Creation**
- Structure information in order of usefulness
- Lead with style references for voice matching
- Follow with specific content requirements
- End with supporting context and details

#### **Eliminate Redundancy**
- Remove duplicate information across sources
- Consolidate similar messaging points
- Focus on unique value from each source

### Smart Processing Rules

#### **For Content Samples** (Preserve completely)
- Include full examples that match {platform}
- Keep original formatting and style
- Maintain exact language and tone
- Prioritize recent, high-quality examples

#### **For Company Documentation** (Summarize intelligently)
- Extract key messaging pillars
- Identify relevant product/service details
- Pull specific facts, numbers, and concrete details
- Condense background info to essential points

#### **For URL Content** (Process strategically)
- Summarize main points relevant to the request
- Extract key facts, quotes, or data points
- Identify connections to {company_name} messaging
- Focus on information that adds value to the content

#### **For Brand Guidelines** (Include key requirements)
- Maintain specific voice requirements
- Keep exact tone and style guidelines
- Preserve banned phrases and quality standards
- Include platform-specific requirements

## Output Structure

Organize the final context as follows:

```
=== CONTENT STYLE REFERENCE ===
[Exact content samples for {platform} - preserve word-for-word]

=== BRAND VOICE REQUIREMENTS ===
[Specific voice guidelines, tone requirements, banned phrases]

=== REQUEST-SPECIFIC CONTEXT ===
[Information directly relevant to "{request}"]

=== CORE COMPANY MESSAGING ===
[Key messaging pillars, value propositions, positioning]

=== SUPPORTING DETAILS ===
[Additional context that enhances content quality]

=== KEY REQUIREMENTS SUMMARY ===
- Platform: {platform}
- Target: {target_audience}
- Word limit: [extract from guidelines]
- Tone: [extract from brand requirements]
- Key focus: [based on request analysis]
```

### Quality Control Guidelines

#### **Content Sample Processing**
- **Never edit or paraphrase** authentic voice examples
- **Include complete examples**, not excerpts
- **Prioritize quality over quantity** - better to have fewer, perfect examples
- **Match platform context** - {platform} samples first

#### **Information Hierarchy**
- **Most relevant first** - what will most improve content quality
- **Authentic voice examples** before company information
- **Specific before general** - concrete details over broad statements
- **Actionable before background** - what helps create content vs. nice-to-know

#### **Context Optimization**
- **Preserve critical details** like numbers, names, specific examples
- **Maintain brand language** exactly as written
- **Remove outdated information** that doesn't reflect current voice
- **Focus on quality indicators** that demonstrate excellence

### Platform-Specific Organization

For **{platform}** content:
- Emphasize {platform}-appropriate voice examples
- Include platform-specific formatting guidelines
- Highlight successful {platform} content patterns
- Focus on {platform} audience expectations

### Final Context Assembly

**Goal**: Create a perfectly organized context that enables the content creator to produce authentic, high-quality {company_name} content that:
- Matches the exact brand voice from samples
- Provides clear value to {target_audience}
- Fits {platform} culture and expectations
- Meets all quality and style requirements

**Success Metric**: The organized context should be so clear and well-structured that it naturally guides the content creator to produce excellent, brand-consistent results.

---
**Organization Date**: {current_date}
**Processing Target**: {platform} content for {company_name}
**Request Focus**: "{request}"
**Available Context**: Company docs, content samples, and URL content