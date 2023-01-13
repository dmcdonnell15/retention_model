
declare @term1 varchar(50) = '2023sp'
declare @term2 varchar(50) = '2022sp'
declare @term3 varchar(50) = '2021sp'
declare @term4 varchar(50) = '2020sp'

select
	t1.[student id]
	, t1.Term [Start Term]
	, [GPA Institutional Cumulative]
	, case when t2.[Student ID] is not null then 1 else 0 end [Retained]
	, case when ([start of term retention status] in ('new', 'transfer in') or [starting cohort term] = t1.term) then 1 else 0 end [new_student]
	, case when tests.[Student ID] is null then 0 else 1 end [ptest_exists]
	, max(aleks) aleks, max(sat_english) sat_english, max(sat_math) sat_math, max(act_composite) act_composite, max(cccrtw) cccrtw
	, case when max(count_tests) is null then 0 else max(count_tests) end count_tests
	, case when waiver.[Student ID] is null then 0 else 1 end test_waiver 
    , t1.[Home College] [Home_College]
	, case when t1.[Home College] = 'DA' then 1 else 0 end [HC_DA]
	, case when t1.[Home College] = 'KK' then 1 else 0 end [HC_KK]
	, case when t1.[Home College] = 'MX' then 1 else 0 end [HC_MX]
	, case when t1.[Home College] = 'OH' then 1 else 0 end [HC_OH]
	, case when t1.[Home College] = 'TR' then 1 else 0 end [HC_TR]
	, case when t1.[Home College] = 'WR' then 1 else 0 end [HC_WR]
	, case when t1.[Home College] not in ('DA', 'HW', 'KK', 'MX', 'OH', 'TR', 'WR') then 1 else 0 end [HC_other]
	, [First_reg]
	, case when t1.[Gender] = 'Male' then 1 else 0 end [Gender_Male]
	, case when t1.[Gender] = '(Blank)' then 1 else 0  end [Gender_Unknown]
	, case when [Age at Census] is null then 0 else [age at census] end [age]
	, case when t1.[Ethnicity] = 'Black' then 1 else 0 end [Eth_Black]
	, case when t1.[Ethnicity] = 'Asian' then 1 else 0 end [Eth_Asian]
	, case when t1.[Ethnicity] = 'White' then 1 else 0 end [Eth_White]
	, case when t1.[Ethnicity] not in ('Black', 'Hispanic', 'White', 'Asian') then 1 else 0 end [Eth_Other]
	, case when t1.[STAR Eligibility] = 'STR' then 1 else 0 end [Star Status]
	, case when [Declared Degree] in ('AC', 'BC') then 1 else 0 end [Deg_cert]
	, case when [Declared Degree] in ('AAS', 'AGS') then 1 else 0 end [Deg_terminal]
	, case when [Full Or Part Time] = 'Full-Time' then 1 else 0 end [FT]
	, case when [Ever Early College] is null then 0 else 1 end [Ever Early College]
	, case when t1.[Athletic Indicator] = 'no' then 0 else 1 end [Athletic Indicator]
	, case when pelig.[Student ID] is null then 0 else 1 end [Pell Eligibility Status]
--	, case when pell_recipient_status = 'Pell Recipient' then 1 else 0 end [pell recipient status]
	, case when (bridg.[student id] is not null 
        or (gwy.[Student ID] is not null 
            and ([start of term retention status] in ('new', 'transfer in') or [starting cohort term] = t1.term)))
        and t1.[home college] = 'TR' then 1 else 0 end [Gateway/Bridge status]

from
	openbook.dbo.pvt_studentterms t1
	
left join /*Exclude completers*/
(select distinct [Student ID], [Term Order]
from openbook.dbo.pvt_studentdegrees) 
degree on t1.[Student ID] = degree.[Student ID] and t1.[Term Order] = degree.[Term Order]

left join /*Check subsequent term enrollment*/
(select [Student ID], [Term Order], term
from openbook.dbo.pvt_StudentTerms
where Enrolled = 'yes' and [Instructional Area] = 'Semester Credit')
t2 on t1.[Student ID] = t2.[Student ID] and t1.[Term Order] = (t2.[Term Order] - 99)

/*Registration Date*/
left join
(select colleagueid, term, min([first_reg]) [First_reg] from (
	(select colleagueid, term, datediff(day, startDate, statusdate) [First_reg] from openbook.dbo.tbl_dropadd da
		left join (select  classsection, startdate from openbook.dbo.tbl_classsections) sd on da.classSection = sd.classSection
	where [status] = 'E' and academicLevel = 'cred' group by [colleagueId], term, startdate, statusdate)) reg2
group by colleagueid, term)
reg on t1.[student id] = reg.colleagueid and t1.term = reg.term

/*Early College*/
left join
(select distinct [student id], [Ever Early College] = 'Was EC' from openbook.dbo.pvt_studentterms where [enrolled] = 'yes' and [Early College] = 'yes') 
ec on t1.[Student ID] = ec.[Student id]

/*Placement test scores and count of tests*/
left join
(select [student id], [SAT - New - Evidence Based Read & Writing] sat_english, [SAT - New - Math Section Score SAT] sat_math, [CCCRTW - Writing] cccrtw
	, [ACT - Composite] act_composite, [ALEKS PPL - Math] aleks, count_tests, [term taken order]
from  
	(select [student id], test, [high score], [term taken order], row_number() over (partition by [student id] order by [term taken order]) count_tests
	from openbook.dbo.pvt_studenttests
	where [test] in ('CCCRTW - Writing', 'ACT - Composite', 'ALEKS PPL - Math') or ([test category] in ('sat') and [high score] >=100))
	sourcetable
	PIVOT
	(max([high score])
	for test IN ([SAT - New - Evidence Based Read & Writing], [SAT - New - Math Section Score SAT], [CCCRTW - Writing], [ACT - Composite], [ALEKS PPL - Math])) pivottable) 
tests on t1.[student id] = tests.[Student ID] and t1.[Term Order] >= tests.[term taken order]

/*Placement test waiver*/
left join
(select [student id], min([term taken order]) [term taken order] from openbook.dbo.pvt_studenttests 
where [test category] = 'placement test waiver' group by [student id])-- and concat(left(@term1, 4), 02) >= [term taken order])
waiver on t1.[student id] = waiver.[Student ID] and t1.[term order] >= waiver.[term taken order]

/*Student Financials: Pell eligible, Pell recipient, checklist items, DEL holds*/
left join
(select distinct [Student ID], [Financial Aid Federal ID], [Financial Aid Year]
from [Openbook].[dbo].[pvt_StudentFinancialAids] 
where  [Financial Aid Federal ID] = 'PELL' and [financial aid pell candidacy] = 'yes') 
pelig on t1.[Student ID] = pelig.[student id] and left(t1.Term, 4) = pelig.[financial aid year]

/*Gateway/Bridge students for Truman*/
left join 
(select distinct [student id] from openbook.dbo.vw_ccc_pvt_ServiceIndicator 
where [Service Indicator Reason] IN ('bridg') and term = @term1)
bridg on t1.[student id] = bridg.[Student ID] 

left join 
(select distinct [student id] from openbook.dbo.vw_ccc_pvt_ServiceIndicator 
where [Service Indicator Reason] IN ('gw1', 'gwy') and term = @term1)
gwy on t1.[student id] = gwy.[Student ID] 

where
	t1.Term in (@term1, @term2, @term3, @term4)
	and Enrolled = 'yes' and [Instructional Area] = 'semester credit' 
	and degree.[Student ID] is null and [Early College] = 'no' and [declared degree] not in ('na')
    and [Academic Plan] not in ('Communications Technology-AAS', 'Electric Construction Tech-AAS')
group by t1.[student id], t2.[Student ID], [start of term retention status], tests.[Student ID], waiver.[Student ID], t1.Term, t1.[Home College]
	, [First_reg], t1.[Gender], [age at census], t1.Ethnicity, t1.[STAR Eligibility], [Declared Degree], [Full or Part Time], [Ever Early College]
	, t1.[Athletic Indicator], pelig.[Student ID], [Starting Cohort Term], [GPA Institutional Term], gwy.[Student ID], bridg.[Student ID], gwy.[Student ID]
    , [gpa institutional cumulative]
order by [student id], [Start Term]
