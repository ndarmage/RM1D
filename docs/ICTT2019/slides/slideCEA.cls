%% %% ==========================================
%% %% Beamer Class defined by JB Blanchard
%% %% jean-baptiste.blanchard@cea.fr
%% %% ==========================================

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%% Class definition %%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slideCEA}[2015/11/20 CEA slide LaTeX class]

%%\LoadClassWithOptions{beamer}
\LoadClass[14pt]{beamer} %% So that the font size remain 14pt (everything has been done with this)
\usetheme[headheight=0.2ex,footheight=0.2ex]{boxes}

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%% Global geometry properties %%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% PAGE SIZE and define math values for
\pgfmathsetmacro\cea@pagewidth{254}
\pgfmathsetmacro\cea@pageheight{190.5}
\pgfmathsetmacro\cea@Marge{-10}
\pgfmathsetmacro\cea@montextwidth{230}
\pgfmathsetmacro\cea@montextheight{170}
\setlength{\paperwidth}{\cea@pagewidth mm}
\setlength{\paperheight}{\cea@pageheight mm}
\setlength{\textwidth}{\cea@montextwidth mm}
\setlength{\textheight}{\cea@montextheight mm}
\setlength{\rightmargin}{0mm}
\setlength{\leftmargin}{0mm}

%% %% Put text wherever you want
\RequirePackage[absolute,showboxes,overlay]{textpos}
\setlength{\TPHorizModule}{1mm}
\setlength{\TPVertModule}{\TPHorizModule}
\textblockorigin{1mm}{1mm} % origine des positions
\TPshowboxesfalse

%% %% macro to enlarge locally the text area
\newenvironment{changemargin}[2]{\begin{list}{}{%
      \setlength{\topsep}{0pt}%
      \setlength{\leftmargin}{0pt}%
      \setlength{\rightmargin}{0pt}%
      \setlength{\listparindent}{\parindent}%
      \setlength{\itemindent}{\parindent}%
      \setlength{\parsep}{0pt plus 1pt}%
      \addtolength{\leftmargin}{#1}%
      \addtolength{\rightmargin}{#2}%
    }\item }{\end{list}}
%% %% end of the macro

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%% Usual package loading %%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% Coding package for french (all are maybe not usefull)
\RequirePackage[T1]{fontenc}  %% %% for specific characters like '|'
\RequirePackage[utf8]{inputenc} %% %% for general accented characters
\RequirePackage{bm} %bold symbol

%% %% If packages
\RequirePackage{etoolbox} %% %% Redefine Block color (\AtBegin)
\RequirePackage{xstring} %% %% (Cut and test string)

%% %% Section counters
%% %% Used to remove backup automatically from main toc
\RequirePackage{totcount}
\newtotcounter{SectionCounter}

%% %% Table option
\PassOptionsToPackage{table}{xcolor}
\RequirePackage{colortbl}
\RequirePackage{multirow}
\RequirePackage{bigstrut}

%% %% Figure options
\RequirePackage{subfigure}
\RequirePackage{epsfig}
\RequirePackage{rotating}

%% %% Using TIKZ packages
\RequirePackage{calc}
\RequirePackage{tikz}
\usetikzlibrary{fadings,positioning,arrows,shapes,calc,shadings,decorations.pathmorphing,decorations.markings}
\tikzstyle{every picture}+=[remember picture]
\tikzstyle{na} = [baseline=-.5ex]

%% %% hyperreference
\RequirePackage{hyperref}

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%% Personnal style modifications %%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% %% Personnal packages: math/physics
%% %% \RequirePackage{personnalphysics}

%% %% Define a shortcut for new frame
\newcommand{\theframe}[2]{
  \begin{frame}
    \frametitle{#1}
    #2
  \end{frame}
}

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%% Global style modifications %%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %% Change font
\renewcommand{\rmdefault}{phv} % Arial
\renewcommand{\sfdefault}{phv} % Arial
\setbeamerfont{frametitle}{%
  family=\rmfamily,series=\bfseries,size=\Large
}
\setbeamerfont{section in toc}{%
  family=\rmfamily,series=\mdseries,size=\Large
}
\setbeamerfont{subsection in toc}{%
  family=\rmfamily,series=\mdseries,size=\large
}
\setbeamerfont{normal text}{%
  family=\rmfamily,series=\mdseries,size=\normalsize
}
\setbeamerfont{text}{%
  family=\rmfamily,series=\mdseries,size=\normalsize
}

%% %% Change the way things look in TOC
\setbeamercolor{section in toc}{bg=white,fg=ceared1}
\setbeamercolor{frametitle}{bg=white,fg=white}
\setbeamercolor{normal text}{bg=white,fg=black!60}
\setbeamertemplate{subsection in toc}{
  ~~~~~\inserttocsubsection\\
}

%% %% Itemize customisation
\setbeamercolor{structure}{fg=ceagreen3}
%% \setbeamertemplate{itemize item}{$\blacksquare$}
%% \setbeamertemplate{itemize subitem}{$\blacksquare$}
%% \setbeamertemplate{itemize subsubitem}{$\blacksquare$}
%\setbeamertemplate{itemize item}{\raisebox{0.5mm}{\includegraphics[width=4.5mm]{petitcarrevert.png}}}
%\setbeamertemplate{itemize subitem}{\raisebox{0.5mm}{\includegraphics[width=4mm]{petitcarrevert.png}}}
%\setbeamertemplate{itemize subsubitem}{\raisebox{0.5mm}{\includegraphics[width=3.5mm]{petitcarrevert.png}}}
\setbeamertemplate{itemize item}{{\includegraphics[width=4.5mm]{petitcarrevert.png}}}
\setbeamertemplate{itemize subitem}{{\includegraphics[width=4mm]{petitcarrevert.png}}}
\setbeamertemplate{itemize subsubitem}{{\includegraphics[width=3.5mm]{petitcarrevert.png}}}

%% %% Block definition:
\setbeamertemplate{blocks}[rounded]
\setbeamerfont{block title}{family=\rmfamily,series=\mdseries,size=\Large}
%% %% Normal block
\setbeamercolor{block title}{fg=ceagreen3,bg=white}
\setbeamercolor{block body}{fg=black!60,bg=white}
%% %% Alert block
\setbeamercolor{block body alerted}{fg=black!60,bg=ceared1!15}
\setbeamercolor{block title alerted}{fg=ceared3,bg=ceared1!15}
\AtBeginEnvironment{alertblock}{
  \setbeamercolor{description item}{fg=ceared3} \setbeamercolor{enumerate item}{fg=ceared3}
  \setbeamercolor{description subitem}{fg=ceared3} \setbeamercolor{enumerate subitem}{fg=ceared3}
  \setbeamercolor{description subsubitem}{fg=ceared3} \setbeamercolor{enumerate subsubitem}{fg=ceared3}
  %% \setbeamercolor{itemize item}{fg=ceared3}
  %% \setbeamercolor{itemize subitem}{fg=ceared3}
  %% \setbeamercolor{itemize subsubitem}{fg=ceared3}
  \setbeamertemplate{itemize item}{\raisebox{-0.5mm}{\includegraphics[width=4.5mm]{petitcarrerouge.png}}}
  \setbeamertemplate{itemize subitem}{ \raisebox{-0.5mm}{\includegraphics[width=4mm]{petitcarrerouge.png}}}
  \setbeamertemplate{itemize subsubitem}{ \raisebox{-0.5mm}{\includegraphics[width=3.5mm]{petitcarrerouge.png}}}

}
%% %% Example block
\setbeamercolor{block body example}{fg=black!60,bg=ceablue1!15}
\setbeamercolor{block title example}{fg=ceablue3,bg=ceablue1!15}
\AtBeginEnvironment{exampleblock}{
  \setbeamercolor{description item}{fg=ceablue3} \setbeamercolor{enumerate item}{fg=ceablue3}
  \setbeamercolor{description subitem}{fg=ceablue3} \setbeamercolor{enumerate subitem}{fg=ceablue3}
  \setbeamercolor{description subsubitem}{fg=ceablue3} \setbeamercolor{enumerate subsubitem}{fg=ceablue3}
  %% \setbeamercolor{itemize item}{fg=ceablue3}
  %% \setbeamercolor{itemize subitem}{fg=ceablue3}
  %% \setbeamercolor{itemize subsubitem}{fg=ceablue3}
  \setbeamertemplate{itemize item}{\raisebox{-0.5mm}{\includegraphics[width=4.5mm]{petitcarrebleu.png}}}
  \setbeamertemplate{itemize subitem}{ \raisebox{-0.5mm}{\includegraphics[width=4mm]{petitcarrebleu.png}}}
  \setbeamertemplate{itemize subsubitem}{ \raisebox{-0.5mm}{\includegraphics[width=3.5mm]{petitcarrebleu.png}}}
}

\DeclareOption{black}{
  \setbeamercolor{normal text}{bg=white,fg=black}
  \setbeamercolor{block body}{bg=white,fg=black}
  \setbeamercolor{block body alerted}{bg=ceared1!15,fg=black}
  \setbeamercolor{block body example}{bg=ceablue1!15,fg=black}
}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%% Define official Style %%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\addfootbox{fg=black,bg=white}{%
  \hspace*{\fill} \raisebox{8mm}{
    \small%
    \cea@BottomBarInstitute~| \hyperlink{Menu<4>}{\insertshortdate} |  \hyperlink{MenuBackup}{PAGE \insertframenumber/\inserttotalframenumber{}\hspace*{13mm}}%
  }%
}%
\newif\ifcea@Official
\cea@Officialtrue

%% %% Customized Topbar
\def\cea@TopBar{}

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%% Define PERSONNAL Style %%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %% Customized bottombar
\DeclareOption{notofficial}{

  \cea@Officialfalse

  \setbeamertemplate{footline}{}

  \DeclareOption{bottombar}{
    \setbeamertemplate{footline}{
      %    \addfootbox{bg=white,fg=black}{%
      \raisebox{1mm}{\small%
        \hyperlink{Menu<4>}{\insertshortdate}}%
      \hspace*{\fill}%
      \ifx\cea@Event\@empty%
      \else{
        \raisebox{1mm}{\small \cea@Event~-~}
      }%
      \fi%
      \raisebox{1mm}{\small \insertshortauthor}%
      \hspace*{\fill}%
      \raisebox{1mm}{\small \hyperlink{MenuBackup}{\insertframenumber{} / \inserttotalframenumber}}%
    }%
  }%

  \setbeamercolor{normal text}{bg=white,fg=black}
  \setbeamercolor{block body}{bg=white,fg=black}
  \setbeamercolor{block body alerted}{bg=ceared1!15,fg=black}
  \setbeamercolor{block body example}{bg=ceablue1!15,fg=black}

  \DeclareOption{topbar}{%
    \def\cea@TopBar{ohoui}%
  }%

}%
\ProcessOptions\relax



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%% New command definition %%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% Personnal command to configure the slides
\newcommand{\MailAddress}[1]{\def\cea@MailAddress{ \textcolor{blue}{#1} }} \MailAddress{}
\newcommand{\Telephone}[1]{ \def\cea@Tel{#1} } \Telephone{(0)1 XX XX XX XX}
\newcommand{\Fax}[1]{ \def\cea@Fax{#1} } \Fax{}
\newcommand{\Direction}[1]{ \def\cea@Direction{#1} } \Direction{Direction DIR}
\newcommand{\Departement}[1]{ \def\cea@Departement{#1} } \Departement{Departement DEP}
\newcommand{\Service}[1]{ \def\cea@Service{#1} } \Service{Service SER}
\newcommand{\Centre}[1]{ \def\cea@Centre{#1} } \Centre{Centre de Saclay}
\newcommand{\CentreAddress}[1]{ \def\cea@CentreAddress{#1} } \CentreAddress{91191 Gif-sur-Yvette Cedex}
\newcommand{\OtherAuthor}[1]{ \def\cea@OtherAuthor{#1} } \OtherAuthor{}
\newcommand{\CeaLogo}[1]{ \def\cea@logo{#1}} \CeaLogo{cea-tr.png}
\newcommand{\BottomBarInstitute}[1]{ \def\cea@BottomBarInstitute{#1} } \BottomBarInstitute{CEA}
\newcommand{\FrontPagePicture}[1]{ \def\cea@FrontPagePicture{#1} } \FrontPagePicture{}
\newcommand{\FrontPagePictureSize}[1]{ \pgfmathsetmacro\cea@FrontPagePictureSize{#1} }\FrontPagePictureSize{70}
\newcommand{\FrontPagePictureX}[1]{ \pgfmathsetmacro\cea@XPictFP{#1} }\FrontPagePictureX{140}
\newcommand{\FrontPagePictureY}[1]{ \pgfmathsetmacro\cea@YPictFP{#1} }\FrontPagePictureY{70}
\newcommand{\FrontPagePictureThree}[1]{ \def\cea@FrontPagePictureThree{#1} } \FrontPagePictureThree{}
\newcommand{\FrontPagePictureThreeSize}[1]{ \pgfmathsetmacro\cea@FrontPagePictureThreeSize{#1} }\FrontPagePictureThreeSize{70}
\newcommand{\FrontPagePictureThreeX}[1]{ \pgfmathsetmacro\cea@XPictThreeFP{#1} }\FrontPagePictureThreeX{140}
\newcommand{\FrontPagePictureThreeY}[1]{ \pgfmathsetmacro\cea@YPictThreeFP{#1} }\FrontPagePictureThreeY{70}
\newcommand{\FrontPagePictureTwo}[1]{ \def\cea@FrontPagePictureTwo{#1} } \FrontPagePictureTwo{}
\newcommand{\FrontPagePictureTwoSize}[1]{ \pgfmathsetmacro\cea@FrontPagePictureTwoSize{#1} }\FrontPagePictureTwoSize{70}
\newcommand{\FrontPagePictureTwoX}[1]{ \pgfmathsetmacro\cea@XPictTwoFP{#1} }\FrontPagePictureTwoX{140}
\newcommand{\FrontPagePictureTwoY}[1]{ \pgfmathsetmacro\cea@YPictTwoFP{#1} }\FrontPagePictureTwoY{70}
\newcommand{\TitleBarPicture}[1]{ \def\cea@TitleBarPicture{#1} } \TitleBarPicture{}
\newcommand{\TitleBarPictureSize}[1]{ \pgfmathsetmacro\cea@TBPictureS{#1} }\TitleBarPictureSize{0}
\newcommand{\TitleBarPictureX}[1]{ \pgfmathsetmacro\cea@TBPictureX{#1} }\TitleBarPictureX{230}
\newcommand{\TitleBarPictureY}[1]{ \pgfmathsetmacro\cea@TBPictureY{#1} }\TitleBarPictureY{0}
\newcommand{\TitleBarPictureTwo}[1]{ \def\cea@TitleBarPictureTwo{#1} } \TitleBarPictureTwo{}
\newcommand{\TitleBarPictureTwoSize}[1]{ \pgfmathsetmacro\cea@TBPictureTwoS{#1} }\TitleBarPictureTwoSize{0}
\newcommand{\TitleBarPictureTwoX}[1]{ \pgfmathsetmacro\cea@TBPictureTwoX{#1} }\TitleBarPictureTwoX{230}
\newcommand{\TitleBarPictureTwoY}[1]{ \pgfmathsetmacro\cea@TBPictureTwoY{#1} }\TitleBarPictureTwoY{0}
\newcommand{\TitleBarPictureThree}[1]{ \def\cea@TitleBarPictureThree{#1} } \TitleBarPictureThree{}
\newcommand{\TitleBarPictureThreeSize}[1]{ \pgfmathsetmacro\cea@TBPictureThreeS{#1} }\TitleBarPictureThreeSize{0}
\newcommand{\TitleBarPictureThreeX}[1]{ \pgfmathsetmacro\cea@TBPictureThreeX{#1} }\TitleBarPictureThreeX{230}
\newcommand{\TitleBarPictureThreeY}[1]{ \pgfmathsetmacro\cea@TBPictureThreeY{#1} }\TitleBarPictureThreeY{0}
\newcommand{\FrontPageSubTitleY}[1]{ \pgfmathsetmacro\cea@YSubTitleFP{#1} }\FrontPageSubTitleY{65}
\newcommand{\FrontPageSubTitleX}[1]{ \pgfmathsetmacro\cea@XSubTitleFP{#1} }\FrontPageSubTitleX{110}
% the following tries to implement a line break for too long titles in titlebar
\newcommand{\FrameTitleS}[1]{ \pgfmathsetmacro\cea@FrameTitleS{#1} }\FrameTitleS{60}
\newcommand{\Event}[1]{ \def\cea@Event{#1} } \Event{}
\newcommand{\Subject}[1]{ \def\cea@Subject{#1} } \Subject{} %% not used yet


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%% Position of many object %%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% Define color useed in official theme, see charte for definition (internal definition)
\definecolor{logored1}{RGB}{230,0,25} \definecolor{logored2}{RGB}{200,0,20}
\definecolor{logored3}{RGB}{135,0,10} \definecolor{logored4}{RGB}{90,0,10}
%% %% Shades of red (1 -> 3 is lighter -> darker)
\definecolor{ceared1}{RGB}{204,9,47} \definecolor{ceared2}{RGB}{172,0,64} \definecolor{ceared3}{RGB}{175,41,46}
%% %% Shades of green (1 -> 4 is lighter -> darker)
\definecolor{ceagreen1}{RGB}{181,191,0} \definecolor{ceagreen2}{RGB}{9,184,0}
\definecolor{ceagreen3}{RGB}{65,118,48} \definecolor{ceagreen4}{RGB}{0,105,62}
%% %% Shades of blue (1 -> 3 is lighter -> darker)
\definecolor{ceablue1}{RGB}{0,149,195} \definecolor{ceablue2}{RGB}{83,129,172} \definecolor{ceablue3}{RGB}{0,80,114}
%% %% Shades of orange (1 -> 2 is lighter -> darker)
\definecolor{ceaorange1}{RGB}{255,111,32} \definecolor{ceaorange2}{RGB}{230,161,88}
%% %% Shades of purple (1 -> 2 is lighter -> darker)
\definecolor{ceapurple1}{RGB}{112,44,106} \definecolor{ceapurple2}{RGB}{144,140,202}
%% %% Shades of pink (1 -> 2 is lighter -> darker)
\definecolor{ceapink1}{RGB}{222,28,133} \definecolor{ceapink2}{RGB}{175,149,163}

%% %% Value to set stuff in Front Page (FP)
\newcommand{\XceaFP}[1]{ \pgfmathsetmacro\cea@XceaFP{#1} }\XceaFP{-5}
\newcommand{\YceaFP}[1]{ \pgfmathsetmacro\cea@YceaFP{#1} }\YceaFP{10}
\newcommand{\SceaFP}[1]{ \pgfmathsetmacro\cea@SceaFP{#1} }\SceaFP{90}  %% X,Y and Size of logo
\newcommand{\XAddrFP}[1]{ \pgfmathsetmacro\cea@XAddrFP{#1} }\XAddrFP{32}
\newcommand{\YAddrFP}[1]{ \pgfmathsetmacro\cea@YAddrFP{#1} }\YAddrFP{167}  %% X,Y of www.cea.fr
\newcommand{\XTitleFP}[1]{ \pgfmathsetmacro\cea@XTitleFP{#1} }\XTitleFP{100}
\newcommand{\YTitleFP}[1]{ \pgfmathsetmacro\cea@YTitleFP{#1} }\YTitleFP{30}  %% X,Y of title
\newcommand{\XInfoFP}[1]{ \pgfmathsetmacro\cea@XInfoFP{#1} }\XInfoFP{110}  %% X of all author/date/mail info
\newcommand{\YAuthFP}[1]{ \pgfmathsetmacro\cea@YAuthFP{#1} }\YAuthFP{140}
\newcommand{\XAuthFP}[1]{ \pgfmathsetmacro\cea@XAuthFP{#1} }\XAuthFP{120}
\newcommand{\YMailFP}[1]{ \pgfmathsetmacro\cea@YMailFP{#1} }\YMailFP{147}
\newcommand{\YEventFP}[1]{ \pgfmathsetmacro\cea@YEventFP{#1} }\YEventFP{155}  %% X,Y of title
%\pgfmathsetmacro\cea@YAuthFP{150}
%\pgfmathsetmacro\cea@YMailFP{157}
%\pgfmathsetmacro\cea@YEventFP{164} %% X,Y of title

%% %% CEA LOGO POSITION AND SIZE IN TITLE BAR (==TB)
\pgfmathsetmacro\cea@XceaTB{-3.5}
\pgfmathsetmacro\cea@YceaTB{-3.5}
\pgfmathsetmacro\cea@SceaTB{32}

%% %% Vertical red band width (FP) and Horizontal red band height (TB)
\pgfmathsetmacro\cea@VBandWidth{90}
\pgfmathsetmacro\cea@HBandHeight{23}

%% %% Y of text to be put on intercalaire
\pgfmathsetmacro\cea@Yinter{52}
\pgfmathsetmacro\cea@Yinterplus{68}

%% %% vertical position of infos line on closing slides (in order). Xs are calculated below
\pgfmathsetmacro\cea@NameY{162}
\pgfmathsetmacro\cea@AddrY{166.5}
\pgfmathsetmacro\cea@TeleY{171}
\pgfmathsetmacro\cea@EpicY{180}

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%% Computed values %%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %% Computing the shift to get the band done with respect to LOGO in FP
\pgfmathsetmacro\cea@XFPBR{\cea@VBandWidth}
\pgfmathsetmacro\cea@YFPBR{\cea@pageheight}

%% %% Computing the  to get the band done with respect to LOGO in TB
\pgfmathsetmacro\cea@XTBBR{\cea@pagewidth-\cea@Marge}
\pgfmathsetmacro\cea@YTBBR{\cea@HBandHeight}
\pgfmathsetmacro\TBtitleX{\cea@HBandHeight/1.5}
\pgfmathsetmacro\TBtitleY{\cea@XceaTB+\cea@SceaTB-\cea@Marge}

%% %% For intercalaire slide
\pgfmathsetmacro\cea@Xinter{1.1*\cea@VBandWidth}
\pgfmathsetmacro\interEND{0.9*\cea@pagewidth}
\pgfmathsetmacro\cea@Sinter{0.85*(\interEND-\cea@VBandWidth)}
\pgfmathsetmacro\cea@Xinterplus{1.2*\cea@VBandWidth}

%% %% Closing names
\pgfmathsetmacro\cea@X{1.045*\cea@VBandWidth}
\pgfmathsetmacro\HierX{1.95*\cea@VBandWidth} %%2.2}

%% %% put the border for frametitle
\newcommand{\FrontPageRedStructure}{%

   %% %% Putting CEA logo
    \begin{textblock}{\cea@SceaFP}(\cea@XceaFP,\cea@YceaFP)
      \tikz { \node (PictFP) {\includegraphics[width=\textwidth]{\cea@logo}};}
    \end{textblock}

    %% %% creating the node to draw the shaded red box
    \begin{textblock}{0}(\cea@Marge,\cea@Marge)  \tikz { \node (TL) {};}  \end{textblock}
    \begin{textblock}{0}(\cea@XFPBR,\cea@YFPBR)  \tikz { \node (BR) {};}  \end{textblock}
    \begin{tikzpicture}[overlay]
      \shade[top color=logored1, bottom color=logored3, shading angle=20] (TL) rectangle (BR);
    \end{tikzpicture}

    %% %% Add on top the CEA web address
    \begin{textblock}{\cea@VBandWidth}(\cea@XAddrFP,\cea@YAddrFP) \tikz { \node[color=white] (addFP) {\small www.cea.fr};}\end{textblock}

}%

%% %% Define the frametitle
\newcommand{\theframetitle}{
  \begin{frame}[plain]

    \FrontPageRedStructure
    %% %% Insert the rest of information
    \begin{changemargin}{\cea@VBandWidth mm}{0cm}

      \begin{textblock}{140}(\cea@XTitleFP,\cea@YTitleFP)
        \begin{center}{\huge \inserttitle \par}\end{center}
      \end{textblock}

      \ifx\cea@subtitle\@empty%
      \else{
        \begin{textblock}{120}(\cea@XSubTitleFP,\cea@YSubTitleFP)
          \begin{center}{ \insertsubtitle \par}\end{center}
         \end{textblock}
      }
      \fi

      \ifx\cea@FrontPagePicture\@empty%
      \else{
        \begin{textblock}{\cea@FrontPagePictureSize}(\cea@XPictFP,\cea@YPictFP)
          \includegraphics[width=\textwidth]{\cea@FrontPagePicture}
        \end{textblock}
      }
      \fi
      \ifx\cea@FrontPagePictureTwo\@empty%
      \else{
        \begin{textblock}{\cea@FrontPagePictureTwoSize}(\cea@XPictTwoFP,\cea@YPictTwoFP)
          \includegraphics[width=\textwidth]{\cea@FrontPagePictureTwo}
        \end{textblock}
      }
      \fi
      \ifx\cea@FrontPagePictureThree\@empty%
      \else{
        \begin{textblock}{\cea@FrontPagePictureThreeSize}(\cea@XPictThreeFP,\cea@YPictThreeFP)
          \includegraphics[width=\textwidth]{\cea@FrontPagePictureThree}
        \end{textblock}
      }
      \fi

      \ifcea@Official{

        \begin{textblock}{120}(\cea@XInfoFP,\cea@YAuthFP)
          \centering
         %{ \small \tikz{ \node (auth) {~};} \insertauthor
          { %\tikz{ \node (auth) {~};} \insertauthor
            \insertauthor%
            \ifx\cea@OtherAuthor\@empty%
            \else{, \cea@OtherAuthor}%
            \fi
          }
        \end{textblock}

        %% \begin{tikzpicture}[overlay]
        %%   \fill[color=ceagreen1] ([xshift=0.2mm,yshift=1.75mm]auth) rectangle ([xshift=-0.2mm,yshift=-1.55mm]auth);
        %% \end{tikzpicture}

      }\else{

        \begin{textblock}{120}(\cea@XInfoFP,\cea@YAuthFP)
          { \large \textbf{\insertauthor}
            \ifx\cea@OtherAuthor\@empty%
            \else{, \cea@OtherAuthor}%
            \fi
          }
        \end{textblock}
      }
      \fi

      \begin{textblock}{120}(\cea@XInfoFP,\cea@YMailFP)
        \centering
        {
          \ifx\cea@MailAddress\@empty%
          \else{ \footnotesize \cea@MailAddress }
          \fi
        }
      \end{textblock}

      \ifcea@Official{

        \begin{textblock}{120}(\cea@XInfoFP,\cea@YEventFP)
          \centering
%          {\small \cea@Event{~}--{~}\insertdate{~}--}
          {\small \cea@Event\\[3pt] --{~}\insertdate{~}--}
        \end{textblock}

      }\else{

        \begin{textblock}{180}(\cea@XInfoFP,\cea@YEventFP)
          { \large \cea@Event \tikz{ \node (separ) {~};}\insertdate}
        \end{textblock}

        \begin{tikzpicture}[overlay]
          \fill[color=ceagreen1] ([xshift=0.4mm,yshift=3.25mm]separ) rectangle ([xshift=-0.4mm,yshift=-3.05mm]separ);
        \end{tikzpicture}

      }
      \fi
    \end{changemargin}

  \end{frame}

}

\pgfmathsetmacro\FrameTitleOffset{\cea@SceaTB+\cea@XceaTB}

%% %% Define the frametitle
\setbeamertemplate{frametitle}{

  \pgfmathsetmacro\FrameTitleRM{\cea@TBPictureS+\cea@TBPictureTwoS+\cea@TBPictureThreeS}
  \pgfmathsetmacro\FrameTitleWidth{\cea@pagewidth-\FrameTitleOffset-\cea@TBPictureS-\cea@TBPictureTwoS-\cea@TBPictureThreeS}

  %% %% Putting CEA logo
  \begin{textblock}{\cea@SceaTB}(\cea@XceaTB,\cea@YceaTB) \tikz { \node (PictTB) {\includegraphics[width=\textwidth]{\cea@logo}};} \end{textblock}
  \ifx\cea@TitleBarPicture\empty%
  \else{
    \begin{textblock}{\cea@TBPictureS}(\cea@TBPictureX,\cea@TBPictureY) \tikz { \node (TBPict) {\includegraphics[width=\textwidth]{\cea@TitleBarPicture}};} \end{textblock}
  }
  \fi
  \ifx\cea@TitleBarPictureTwo\empty%
  \else{
    \begin{textblock}{\cea@TBPictureTwoS}(\cea@TBPictureTwoX,\cea@TBPictureTwoY) \tikz { \node (TBPict) {\includegraphics[width=\textwidth]{\cea@TitleBarPictureTwo}};} \end{textblock}
  }
  \fi
  \ifx\cea@TitleBarPictureThree\empty%
  \else{
    \begin{textblock}{\cea@TBPictureThreeS}(\cea@TBPictureThreeX,\cea@TBPictureThreeY) \tikz { \node (TBPict) {\includegraphics[width=\textwidth]{\cea@TitleBarPictureThree}};} \end{textblock}
  }
  \fi

  %% %% creating the node to draw the shaded red box
  \begin{textblock}{0}(\cea@Marge,\cea@Marge)  \tikz { \node (TL) {};}  \end{textblock}
  \begin{textblock}{0}(\cea@XTBBR,\cea@YTBBR)  \tikz { \node (BR) {};}  \end{textblock}
  \begin{tikzpicture}[overlay]
    \shade[left color=logored1, right color=logored3] (TL) rectangle (BR);
  \end{tikzpicture}

  %% %% Cutting the slide title if greater than \cea@FrameTitleS characters
  \expandarg
  \StrLen{\beamer@frametitle}[\titlelen]
  \ifthenelse{\titlelen > \cea@FrameTitleS}{
    \vspace*{2mm}
    \begin{changemargin}{\FrameTitleOffset mm}{\FrameTitleRM mm}
      \insertframetitle \vfill
    \end{changemargin}
   }{
     \vspace*{8mm} \hspace*{\FrameTitleOffset mm} \insertframetitle
   }

  %% %% Adding top bar if requested
  \ifx\cea@TopBar\@empty%
  \else{
    \begin{textblock}{200}(0,0)
      \textcolor{white}{ \footnotesize \insertsection}%
      \ifx\insertsubsection\@empty%
      \else{
        \textcolor{white}{\footnotesize: \insertsubsection}
      }%
      \fi
    \end{textblock}%
  }%
  \fi

}

%% %% Define the red part of both transition and conclusion slides
\newcommand{\transitionRed}{%
  \begin{textblock}{0}(\cea@VBandWidth,\cea@Marge) \tikz { \node (TLIT) {};} \end{textblock}
  \begin{textblock}{0}(\cea@pagewidth,\cea@pageheight) \tikz { \node (BRIT) {};} \end{textblock}
  \begin{tikzpicture}[overlay]
    %% \shade[top color=logored1, bottom color=logored3, shading angle=5] (TLIT) rectangle (BRIT);
    \shade[top color=logored1, bottom color=logored3, shading angle=20] (TLIT) rectangle (BRIT);
  \end{tikzpicture}
}%

%% %% Define the content of the closing frame following CEA example
\newcommand{\theclosingframe}[1]{
  \begin{frame}[plain]
    \transitionRed

    \begin{textblock}{80}(\cea@X,\cea@NameY)
      {\scriptsize \textcolor{white}{Commissariat à l'énergie atomique et aux énergies alternatives}}
    \end{textblock}
    \begin{textblock}{80}(\HierX,\cea@NameY)
      {\scriptsize \textcolor{ceagreen1}{ \cea@Direction}}
    \end{textblock}
    \begin{textblock}{80}(\cea@X,\cea@AddrY)
      {\scriptsize \textcolor{white}{ \cea@Centre \tikz { \node (addr) {~};} \cea@CentreAddress } }
    \end{textblock}
    \begin{textblock}{80}(\HierX,\cea@AddrY)
      {\scriptsize \textcolor{ceagreen1}{ \cea@Departement}}
    \end{textblock}
    \begin{textblock}{80}(\cea@X,\cea@TeleY)
      {\scriptsize \textcolor{white}{ T. +33 \cea@Tel ~\tikz{ \node (tel) {~};}\ifx\cea@Fax\@empty{}\else{ F. +33 \cea@Fax}\fi}}
    \end{textblock}
    \begin{textblock}{75}(\HierX,\cea@TeleY)
      {\scriptsize \textcolor{ceagreen1}{ \cea@Service}}
    \end{textblock}
    \begin{textblock}{90}(\cea@X,\cea@EpicY)
      {\tiny \textcolor{white}{
          Etablissement public à caractère industriel et commercial \tikz{ \node (epic) {~};}  R.C.S Paris B 775 685 019
      }}
    \end{textblock}

    \begin{textblock}{\cea@Sinter}(\cea@Xinter,\cea@Yinter)
      \tikz{ \node[color=white,text width =\cea@Sinter mm, scale=1.25]{\large \bf #1};}
    \end{textblock}

    \begin{tikzpicture}[overlay]
      \fill[color=ceagreen1] ([xshift=0.2mm,yshift=1.75mm]addr) rectangle ([xshift=-0.2mm,yshift=-1.55mm]addr);
      \fill[color=ceagreen1] ([xshift=0.15mm,yshift=1.5mm]epic) rectangle ([xshift=-0.15mm,yshift=-1.3mm]epic);
      \ifx\cea@Fax\@empty{}
      \else{
        \fill[color=ceagreen1] ([xshift=-0.2mm,yshift=1.75mm]tel) rectangle ([xshift=0.2mm,yshift=-1.55mm]tel);
      }
      \fi
    \end{tikzpicture}


  \end{frame}
}

%% %% Define transition with a name following the CEA example
\newcommand{\intercalaire}[1]{
  \begin{frame}[plain]
    \transitionRed
    \begin{textblock}{\cea@Sinter}(\cea@Xinter,\cea@Yinter)
      \tikz{ \node[color=white,text width =\cea@Sinter mm, scale=1.25]{\large \bf #1};}
    \end{textblock}
  \end{frame}
}

%% %% Define transition with a name following the CEA example
\newcommand{\intercalaireplus}[2]{
  \begin{frame}[plain]
    \transitionRed
    \begin{textblock}{\cea@Sinter}(\cea@Xinter,\cea@Yinter)
      \tikz{ \node[color=white,text width =\cea@Sinter mm, scale=1.25]{\large \bf #1};}
    \end{textblock}
    \begin{textblock}{\cea@Sinter}(\cea@Xinterplus,\cea@Yinterplus)
      \tikz{ \node[color=white,text width =\cea@Sinter mm, scale=1.05]{ #2};}
    \end{textblock}
  \end{frame}
}

%% %% Get rid of navigation
\setbeamertemplate{navigation symbols}{}

%% %% Fine tune the couting to get rid of backup in slides counter and sections counter
\newcommand{\backupbegin}{
  \newcounter{framenumberappendix}
  \setcounter{framenumberappendix}{\value{framenumber}}
}
\newcommand{\backupend}{
  \addtocounter{framenumberappendix}{-\value{framenumber}}
  \addtocounter{framenumber}{\value{framenumberappendix}}
}

%\addtobeamertemplate{block begin}{\setlength\abovedisplayskip{15mm}}
