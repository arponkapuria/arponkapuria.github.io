---
title: AWS Nuke - The AWS Resources Destroyer
description: Hands on with a tool designed to delete all resources in an AWS account in one go.

date: May 02, 2026
modified: May 02, 2026

author: Arpon Kapuria
category: Dev Journal
tags: AWS, Tooling, Infrastructure
---


## Background

When working with AWS, especially during experimentation, you quickly accumulate resources—EC2 instances, S3 buckets, IAM roles, VPCs, etc. These can lead to unexpected billing and clutter. There is no one button solution from AWS to remove all these resources. You need to do it manually.

`aws-nuke` is an open-source tool designed to **delete all resources in an AWS account**, effectively resetting it to a clean state. It is extremely powerful and dangerous if misconfigured. 

- It scans your AWS account
- It identifies all resources
- It deletes everything **except what you explicitly protect**

Think of it as:

> “Reset my AWS account to zero, except the things I whitelist.”

## Key concepts

Although most of the people understand these concepts, but still iterating them for those who don’t.

#### 1. Account vs Profile

- *Account ID* → AWS account (e.g., `458973149831`)
- *Profile* → Local AWS CLI credentials

> AWS Nuke uses your **profile to authenticate**, not the account ID directly. There are other ways to authenticate with AWS CLI. You can find them `aws-nuke`’s official documentation.

#### 2. Filters (Critical Safety Mechanism)

Filters define what should **NOT be deleted**. Without filters → you can lock yourself out.


#### 3. Dry Run vs Real Run

- `-dry-run` → shows what *would* be deleted.
- `-no-dry-run` → actually deletes

> Always dry run first.

## Installation

To use `aws-nuke`, we need to configure aws cli first. We need to create an IAM user with an access key in a particular region. This setup is shown for macos.

### 3.1 AWS CLI Setup (Mandatory First Step)

Configure AWS CLI:

```bash
# run this command to install aws-cli
brew install aws-cli

# Configure the aws profiles you would like to use with this cli
aws configure
```

Enter:
- Access Key
- Secret Key
- Region (e.g., `us-east-1`)

Verify:

```bash
# this will show the account connected 
# the profile is configured as 'default' if any profile name is not provided
aws sts get-caller-identity
```

Expected:

```bash
# we will be using 'aws-nuke-example' as the user

{
  "Account":"458973149831",
  "Arn":"arn:aws:iam::458973149831:user/aws-nuke-example"
}
```

### 3.2 AWS Nuke Installation

There are different ways to install `aws-nuke`, such as `github-release`, `Homebrew tap`, `Homebrew core`. We will be using `Homebrew tap` (macos) as it’s directly maintained by the author.

```bash
brew install ekristen/tap/aws-nuke
```

Verify:

```bash
aws-nuke --version
```


### 3.3 Configuration file (Config.yaml)

This is the most important part. Configuration file contains everything. It tells `aws-nuke` what to delete and what not to. There are a lot of filters we can use to keep resources/roles from deleting. This file needs to configured properly.

#### Minimal Safe Config

```yaml
# aws-nuke/config.yaml

regions:
  - global
  - us-east-1
  - eu-north-1

blocklist:
  - "111111111111"  # dummy safety account

accounts:
  "458973149831":
    filters:
      IAMUser:
        - "aws-nuke-example"
      IAMUserPolicyAttachment:
        - "aws-nuke-example -> *"
      IAMUserAccessKey:
        - "aws-nuke-example -> *"
```

#### Explanation

```yaml
regions:
  - global
  - us-east-1
```

(regions) Defines where to delete resources.

- `global` → IAM, Route53, etc.
- Others → region-specific services


```yaml
blocklist:
  -"111111111111"
```

(blocklist) Prevents accidental deletion of critical accounts. Always include something here.

```yaml
accounts:
  "458973149831":
```

(accounts) Target account to clean.

#### Filters (MOST IMPORTANT)

```yaml
IAMUser:
  -"aws-nuke-example"
```

(Protect IAM User) Prevents deletion of your login user.

```yaml
IAMUserPolicyAttachment:
  -"aws-nuke-example -> *"
```

(Protect Policies) Prevents losing permissions (like AdministratorAccess). If you want to something specific to keep, then write the policy instead of **‘*’**.

```yaml
IAMUserAccessKey:
  -"aws-nuke-example -> *"
```

(Protect Access Keys) Prevents CLI access from breaking.

*Extra: Resource-Specific Filters*

```yaml
S3Bucket:
  -"bucket-name"
```

There are more filters we can use for more control over the resources. Refer to the official docs. If you don’t include them:
- Your IAM user gets deleted ❌
- Your access keys get deleted ❌
- You lose access permanently ❌

## Running AWS Nuke

### 4.1 Dry Run from terminal

```bash
aws-nuke nuke -c aws-nuke/config.yaml --profile aws-nuke-example --dry-run

or

aws-nuke nuke -c aws-nuke/config.yaml --profile aws-nuke-example
```

Check:
- IAM user is NOT listed
- Access keys are NOT listed

### 4.2 Actual Execution

```bash
aws-nuke nuke -c aws-nuke/config.yaml --profile aws-nuke-example --no-dry-run
```

## Common Errors & Fixes

#### ❌ Error: No account alias

```bash
# terminal output
specified account doesn't have an alias
```

Fix:
Go to AWS → IAM Dashboard → Create alias. You can set alias  from terminal as well.

```bash
# alias name ex. 'aws-nuke-dev'
aws iam create-account-alias--account-alias aws-nuke-dev
```

#### ❌ Error: Profile not found

```
The config profile could not be found
```

Fix:
It’s mostly because either you’re not providing the correct profile or the profile is not configured. Check profiles:

```bash
aws configure list-profiles
```

If no profile is configured then you can use `default` :

```bash
--profile default
```

#### ❌ Error: Access lost after nuke

Cause:
- IAMUser not protected
- AccessKey not protected

Fix: Recreate user via root login

## Resources Deleted / Not Deleted

AWS Nuke removes:

- EC2 instances
- S3 buckets
- RDS databases
- VPCs
- Lambda functions
- CloudWatch logs
- IAM roles (if not filtered)

What Does NOT Get Deleted

- Account itself
- Billing info
- Root user

## Extras

AWS Nuke is one of the most powerful tools in AWS infrastructure management.

Used correctly, it:
- Saves cost
- Maintains clean environments
- Enables rapid experimentation

Used incorrectly, it:
- Locks you out
- Deletes critical infrastructure

**Best Practices**

1. *Use Separate Accounts*
    - Dev account → safe to nuke
    - Prod account → NEVER nuke
2. *Always Dry Run*
    
    Never skip this:
    
    ```bash
    --dry-run
    ```
    
3. *Protect Access Layer*
    
    Always include:
    
    - IAMUser
    - IAMUserPolicyAttachment
    - IAMUserAccessKey
4. *Start Small*
    
    Initially restrict regions:
    
    ```yaml
    regions:
      - us-east-1
    ```
    
5. *Use for Reset Workflows*
    
    Ideal use cases:
    
    - Cleaning dev environments
    - Resetting experiments
    - Cost control

<hr>

## References

1. https://aws-nuke.ekristen.dev/ - *Official docs*
2. https://github.com/ekristen/aws-nuke - *Github*