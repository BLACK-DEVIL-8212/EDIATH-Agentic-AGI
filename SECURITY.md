# Security Policy

## EDIATH Agentic AI Security Policy

At EDIATH Agentic AI, security is a top priority. We appreciate the efforts of security researchers and the community in helping us identify and responsibly disclose vulnerabilities.

---

## Supported Versions

The following versions are currently supported with security updates:

| Version | Supported |
| ------- | --------- |
| 1.x.x   | ✅ Yes     |
| 0.x.x   | ❌ No      |

Only the latest stable release receives security patches and updates.

---

## Reporting a Vulnerability

If you discover a security vulnerability within EDIATH Agentic AI, please report it responsibly.

### Contact Information

Security reports can be submitted through:

* Email: [security@ediath.ai](shakshamshakshamsingh@gmail.com)
* Alternate Email: [support@ediath.ai](shakshamshakshamsingh@gmail.com)

### Information to Include

Please provide:

* Detailed description of the vulnerability.
* Steps to reproduce the issue.
* Impact assessment.
* Proof of Concept (PoC) if available.
* Affected version(s).
* Screenshots or logs if applicable.

### Response Timeline

| Stage                     | Expected Time     |
| ------------------------- | ----------------- |
| Initial Acknowledgement   | Within 48 Hours   |
| Investigation             | Within 7 Days     |
| Status Update             | Every 7 Days      |
| Resolution (if confirmed) | Based on Severity |

---

## Responsible Disclosure Guidelines

Please:

* Do not publicly disclose the vulnerability before it has been investigated and resolved.
* Do not access, modify, or delete data belonging to others.
* Do not perform denial-of-service attacks.
* Do not exploit vulnerabilities beyond what is necessary to demonstrate their existence.

---

## Vulnerability Severity Levels

### Critical

* Remote Code Execution (RCE)
* Authentication Bypass
* Privilege Escalation
* Full System Compromise

### High

* Sensitive Data Exposure
* Agent Takeover
* API Key Exposure
* Arbitrary File Access

### Medium

* Information Disclosure
* Cross-Site Scripting (XSS)
* CSRF Vulnerabilities
* Misconfigured Access Controls

### Low

* Minor Security Misconfigurations
* Non-Sensitive Information Leakage

---

## Security Features

EDIATH Agentic AI implements:

* Role-Based Access Control (RBAC)
* API Authentication
* Session Security Controls
* Secure Credential Storage
* Audit Logging
* Encryption in Transit (TLS)
* Environment Variable Protection
* Rate Limiting
* Agent Isolation Mechanisms
* Secure Model Execution Pipelines

---

## Third-Party Dependencies

EDIATH Agentic AI relies on third-party libraries and AI frameworks. Security updates should be applied regularly to:

* Python Packages
* Node.js Packages
* AI Frameworks
* Databases
* Operating System Components

---

## Security Update Policy

Security fixes are prioritized based on:

1. Exploitability
2. Potential Impact
3. Exposure Surface
4. Availability of Mitigations

Critical vulnerabilities may be patched immediately and released outside the normal release schedule.

---

## Recognition

We thank responsible security researchers who help improve the security of EDIATH Agentic AI.

Contributors who responsibly disclose valid security vulnerabilities may be acknowledged in future security advisories unless anonymity is requested.

---

Last Updated: May 2026
Project: EDIATH Agentic AI

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.
